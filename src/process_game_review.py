#!usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from helpers import save_csv


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = BASE_DIR / "tables" / "raw" / "all_reviews.csv"
DEFAULT_OUTPUT_FILE = BASE_DIR / "tables" / "production" / "GAME_REVIEW.csv"
GAME_DATA_FILE = BASE_DIR / "tables" / "production" / "GAME_DATA.csv"


READ_CHUNK_SIZE = 1_000_000

RAW_COLUMNS = {
    "recommendationid": "review_id",
    "appid": "item_id",
    "author_steamid": "steam_id",
    "author_playtime_forever": "hours",
    "author_playtime_at_review": "hours_at_review",
    "timestamp_created": "date_created",
    "voted_up": "recommendation",
    "written_during_early_access": "early_access",
}

def load_game_median_playtime(input_path: Path = GAME_DATA_FILE) -> pd.DataFrame:
    game_df = pd.read_csv(input_path, usecols=["item_id", "median_playtime_forever"])
    game_df["item_id"] = pd.to_numeric(game_df["item_id"], errors="coerce")
    game_df["median_playtime_forever"] = pd.to_numeric(game_df["median_playtime_forever"],errors="coerce").fillna(0.0)
    return game_df.dropna(subset=["item_id"]).copy()


# assign recommendation based on normalized hours compared to median playtime of the game:
# 0: not recommended, 1: neutral, 2: recommended
# the threshold values are based on EDA analysis in noteboooks/recommendation_normalized_hours_eda.ipynb
def assign_recommendation_from_normalized_hours(normalized_hours: pd.Series) -> pd.Series:
    recommendation = pd.Series(1, index=normalized_hours.index, dtype="int8")
    recommendation = recommendation.mask(normalized_hours < 0.1, 0)
    recommendation = recommendation.mask(normalized_hours > 0.3, 2)
    return recommendation.astype("int8")


def load_arhive_reviews(input_path: Path) -> pd.DataFrame:
    game_df = load_game_median_playtime()

    dtype_map = {
        "recommendationid": "Int64",
        "appid": "Int64",
        "author_steamid": "Int64",
        "author_playtime_forever": "float32",
        "author_playtime_at_review": "float32",
        "timestamp_created": "Int64",
        "voted_up": "Int8",
        "written_during_early_access": "Int8",
    }

    chunk_deduped: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        input_path,
        usecols=list(RAW_COLUMNS),
        dtype=dtype_map,
        chunksize=READ_CHUNK_SIZE,
    ):
        chunk = chunk.rename(columns=RAW_COLUMNS)

        chunk["date_created"] = pd.to_numeric(chunk["date_created"], errors="coerce")
        chunk["hours"] = pd.to_numeric(chunk["hours"], errors="coerce").fillna(0.0)
        chunk["hours_at_review"] = pd.to_numeric(chunk["hours_at_review"], errors="coerce").fillna(0.0)
        chunk["recommendation"] = pd.to_numeric(chunk["recommendation"], errors="coerce").fillna(0).astype("int8")
        chunk["early_access"] = pd.to_numeric(chunk["early_access"], errors="coerce").fillna(0).astype("int8")

        chunk = chunk[chunk["hours_at_review"] > 2].copy() # only keep hours at review more than 2 hours -> reliable data
        if chunk.empty:
            continue

        chunk = chunk.merge(game_df, on="item_id", how="left")
        median_hours = chunk["median_playtime_forever"].fillna(0.0).clip(lower=0.0)
        normalized_hours = pd.Series(0.0, index=chunk.index, dtype="float32")
        valid_mask = median_hours >= 1.0 # only normalize for games with median playtime more than 1 hour to avoid inflating normalized hours for very short games
        normalized_hours.loc[valid_mask] = (chunk.loc[valid_mask, "hours"].astype("float32") / median_hours.loc[valid_mask].astype("float32")) # normalized by game's median playtime
        chunk["recommendation"] = assign_recommendation_from_normalized_hours(normalized_hours)

        chunk_deduped.append(chunk[[
            "review_id", "item_id", "steam_id", "hours", "date_created", "recommendation", "early_access"
            ]])

    if not chunk_deduped:
        return pd.DataFrame(columns=list(RAW_COLUMNS.values()))

    df = pd.concat(chunk_deduped, ignore_index=True)
    del chunk_deduped


    # dup_count = df.duplicated(subset=["steam_id", "item_id"]).sum() # -> ~2400 rows out of 110M
    df = df.sort_values(["steam_id", "item_id", "date_created"], ascending=[True, True, False]) 
    df = df.drop_duplicates(subset=["steam_id", "item_id"], keep="first").reset_index(drop=True) # keep the most recent review for each user-item pair

    # timestamp in Unix time seconds, convert to date string
    df["date_created"] = pd.to_datetime(df["date_created"], unit="s", utc=True).dt.strftime("%Y-%m-%d")


    return df

def main() -> None:
    
    reviews_df = load_arhive_reviews(DEFAULT_INPUT_FILE)
    save_csv(reviews_df, DEFAULT_OUTPUT_FILE)
    print(f"[csv] Wrote {DEFAULT_OUTPUT_FILE} ({len(reviews_df):,} rows)")

if __name__ == "__main__":
    main()
