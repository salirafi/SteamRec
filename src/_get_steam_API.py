import json
import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import requests

API_KEY = os.getenv("STEAM_WEB_API_KEY", "")

url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"


def fetch_owned_games_payload(steam_id: int | str, api_key: str = None) -> dict:
    resolved_api_key = api_key or os.getenv("STEAM_WEB_API_KEY") or API_KEY
    payload = {
        "steamid": str(steam_id),
        "include_appinfo": True,
        "include_played_free_games": True
    }

    r = requests.get(
        url,
        params={
            "key": resolved_api_key,
            "input_json": json.dumps(payload),
            "format": "json",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def owned_games_payload_to_df(data: dict) -> pd.DataFrame:
    games = data.get("response", {}).get("games", [])
    if not games:
        return pd.DataFrame(columns=["item_id", "hours", "recommend"])

    interactions_df = pd.DataFrame(games).reindex(columns=["appid", "playtime_forever"]).copy()
    interactions_df["appid"] = pd.to_numeric(interactions_df["appid"], errors="coerce")
    interactions_df["playtime_forever"] = pd.to_numeric(interactions_df["playtime_forever"],errors="coerce").fillna(0.0)
    interactions_df = interactions_df.dropna(subset=["appid"]).copy()
    if interactions_df.empty:
        return pd.DataFrame(columns=["item_id", "hours", "recommend"])

    interactions_df["item_id"] = interactions_df["appid"].astype("int64")
    interactions_df["hours"] = (interactions_df["playtime_forever"] / 60.0).astype("float32")
    interactions_df["recommend"] = int(1)
    return interactions_df[["item_id", "hours", "recommend"]].reset_index(drop=True)


def fetch_owned_games_df(steam_id: int | str, api_key: str | None = None) -> pd.DataFrame:
    data = fetch_owned_games_payload(steam_id=steam_id, api_key=api_key)
    return owned_games_payload_to_df(data)


def main() -> None: # test
    STEAM_ID = "76561198272974330"
    data = fetch_owned_games_payload(steam_id=STEAM_ID)
    print("FULL JSON RESPONSE:")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
