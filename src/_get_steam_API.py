#!/usr/bin/env python3

import json
import os

from dotenv import load_dotenv
import pandas as pd
import requests
from sqlalchemy import bindparam, create_engine, text

try:
    from src.config import DB_CONFIG, get_db_url
except ImportError:
    from config import DB_CONFIG, get_db_url

load_dotenv()

API_KEY = os.getenv("STEAM_WEB_API_KEY", "")

OWNED_GAMES_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
DEFAULT_TIMEOUT = 30


def fetch_owned_games_payload(steam_id: int | str, api_key: str | None = None) -> dict:
    resolved_api_key = api_key or os.getenv("STEAM_WEB_API_KEY") or API_KEY
    payload = {
        "steamid": str(steam_id),
        "include_appinfo": True,
        "include_played_free_games": True,
    }

    response = requests.get(
        OWNED_GAMES_URL,
        params={
            "key": resolved_api_key,
            "input_json": json.dumps(payload),
            "format": "json",
        },
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def owned_games_payload_to_df(data: dict) -> pd.DataFrame:
    games = data.get("response", {}).get("games", [])
    if not games:
        return pd.DataFrame(columns=["item_id", "hours"])

    interactions_df = pd.DataFrame(games).reindex(columns=["appid", "playtime_forever"]).copy()
    interactions_df["appid"] = pd.to_numeric(interactions_df["appid"], errors="coerce")
    interactions_df["playtime_forever"] = pd.to_numeric(
        interactions_df["playtime_forever"],
        errors="coerce",
    ).fillna(0.0)
    interactions_df = interactions_df.dropna(subset=["appid"]).copy()
    if interactions_df.empty:
        return pd.DataFrame(columns=["item_id", "hours"])

    interactions_df["item_id"] = interactions_df["appid"].astype("int64")
    interactions_df["hours"] = (interactions_df["playtime_forever"]).astype("float32")
    return (
        interactions_df[["item_id", "hours"]]
        .sort_values(["hours", "item_id"], ascending=[False, True])
        .drop_duplicates(subset=["item_id"])
        .reset_index(drop=True)
    )


def fetch_item_median_playtime(engine, item_id_list: list[int]) -> pd.DataFrame:
    if not item_id_list:
        return pd.DataFrame(columns=["item_id", "median_playtime_forever"])

    query = text(
        f"""
        SELECT
            g.item_id,
            g.median_playtime_forever
        FROM {DB_CONFIG["database"]}.GAME_DATA g
        WHERE g.item_id IN :item_id_list
        """
    ).bindparams(bindparam("item_id_list", expanding=True))

    result = pd.read_sql(
        query,
        engine,
        params={"item_id_list": [int(x) for x in item_id_list]},
    )
    if result.empty:
        return pd.DataFrame(columns=["item_id", "median_playtime_forever"])

    result["item_id"] = pd.to_numeric(result["item_id"], errors="coerce").astype("int32")
    result["median_playtime_forever"] = pd.to_numeric(result["median_playtime_forever"],errors="coerce").fillna(0.0).astype("float32")
    return result[["item_id", "median_playtime_forever"]]


def assign_recommendation_from_normalized_hours(normalized_hours: pd.Series) -> pd.Series:
    recommendation = pd.Series(1, index=normalized_hours.index, dtype="int8")
    recommendation = recommendation.mask(normalized_hours < 0.1, 0)
    recommendation = recommendation.mask(normalized_hours > 0.3, 2)
    return recommendation.astype("int8")


def fetch_owned_games_df(
    steam_id: int | str,
    api_key: str | None = None,
    engine=None,
) -> pd.DataFrame:
    owned_games_payload = fetch_owned_games_payload(steam_id=steam_id, api_key=api_key)
    owned_games_df = owned_games_payload_to_df(owned_games_payload)
    if owned_games_df.empty:
        return pd.DataFrame(columns=["item_id", "recommendation", "hours", "early_access"])

    resolved_engine = engine or create_engine(get_db_url())
    game_df = fetch_item_median_playtime(
        resolved_engine,
        owned_games_df["item_id"].astype(int).tolist(),
    )
    result = owned_games_df.merge(game_df, on="item_id", how="left")

    # Per your chosen rule, normalize using the raw median_playtime_forever values from GAME_DATA.
    median_hours = result["median_playtime_forever"].clip(lower=0.0)
    normalized_hours = pd.Series(0.0, index=result.index, dtype="float32")
    valid_mask = median_hours >= 1.0 # only normalize for games with median playtime more than 1 hour to avoid inflating normalized hours for very short games
    normalized_hours.loc[valid_mask] = (result.loc[valid_mask, "hours"].astype("float32") / median_hours.loc[valid_mask].astype("float32"))

    result["recommendation"] = assign_recommendation_from_normalized_hours(normalized_hours)

    # owned-games API does not provide early-access, so use default to 0 for all games
    result["early_access"] = 0

    result["item_id"] = pd.to_numeric(result["item_id"], errors="coerce").astype("int32")
    result["hours"] = pd.to_numeric(result["hours"], errors="coerce").fillna(0.0).astype("float32")
    result["recommendation"] = pd.to_numeric(result["recommendation"], errors="coerce").fillna(1).astype("int8")
    result["early_access"] = pd.to_numeric(result["early_access"], errors="coerce").fillna(0).astype("int8")

    return result[["item_id", "recommendation", "hours", "early_access"]].reset_index(drop=True)


def main() -> None: # test
    steam_id = "76561198272974330"
    interactions_df = fetch_owned_games_df(steam_id=steam_id)
    print(interactions_df.head())
    print(f"Fetched {len(interactions_df)} owned games.")


if __name__ == "__main__":
    main()
