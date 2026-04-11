#!usr/bin/env python3

from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Iterable
import unicodedata

import numpy as np
import pandas as pd

from helpers import sanitize_df, rating_label, save_csv


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = BASE_DIR / "tables" / "raw" / "games.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "tables" / "production"



# take only relevant columns
RAW_COLUMNS = {
    "AppID": "item_id",
    "name": "item_name",
    "release_date": "release_date",
    "estimated_owners": "owners_count",
    "price": "price",
    "median_playtime_forever": "median_playtime_forever",
    "recommendations": "user_reviews",
    "positive": "positive",
    "negative": "negative",
    "developers": "developers",
    "publishers": "publishers",
    "categories": "categories",
    "genres": "genres",
    "tags": "tags",
}

OUTPUT_FILENAMES = {
    "GAME_DATA": "GAME_DATA.csv",
}

INVALID_TEXT_MARKERS = {"", "\\N", "<NA>", "NA"} # all possible NaN markers (case-insnesitive)



def load_archive_games(input_path: Path) -> pd.DataFrame:
    with input_path.open("r", encoding="utf-8-sig") as handle:
        raw_games = json.load(handle)

    games_df = pd.DataFrame.from_dict(raw_games, orient="index")
    del raw_games
    gc.collect()

    games_df.index.name = "AppID"
    games_df = games_df.reset_index()


    selected_df = games_df.loc[:, RAW_COLUMNS.keys()].rename(columns=RAW_COLUMNS).copy()
    del games_df
    gc.collect()
    return selected_df



def parse_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").astype("string") # format "YYYY-MM-DD" as string


# for proper deduplication using lowercase
def normalize_for_dedupe(value: object) -> str:
    if pd.isna(value):
        return ""
    
    # lowercase and collapse whitespace
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip().casefold()

    # remove accents/diacritics
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def normalize_text_series(series: pd.Series, *, strip_na_literal: bool = True) -> pd.Series:
    string_series = series.astype("string").str.strip()
    invalid_markers = INVALID_TEXT_MARKERS if strip_na_literal else INVALID_TEXT_MARKERS - {"NA"}
    return string_series.mask(string_series.isin(invalid_markers), pd.NA) # if value is in invalid markers, set to pd.NA


def normalize_multi_value_text(value: object) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("|", "/") # replace pipe with forward slash to avoid conflict with pipe separator for multi-value fields
    text = re.sub(r"\s+", " ", text).strip() # collapse multiple whitespace into single space and trim leading/trailing whitespace
    return text

# handles tag, genre, category, publisher, and developer fields
def normalize_multi_value_field(value: object) -> list[str]:
    if value is None:
        return []

    # both dicts and lists are possible, including string for "[...]"
    if isinstance(value, dict):
        raw_values = list(value.keys())
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    elif isinstance(value, str):
        raw_values = [value]
    else:
        return []

    cleaned_values: list[str] = []
    for raw_value in raw_values:
        if raw_value is None:
            continue
        if isinstance(raw_value, float) and np.isnan(raw_value):
            continue
        if isinstance(raw_value, (dict, list, tuple, set)):
            continue

        text = normalize_multi_value_text(raw_value)
        if text in INVALID_TEXT_MARKERS:
            continue
        cleaned_values.append(text)

    return list(dict.fromkeys(cleaned_values)) # dedupe while preserving order


def serialize_multi_value_field(value: object) -> str:
    values = normalize_multi_value_field(value)
    if not values:
        return ""
    return "|".join(values) # use pipe as separator for multi-value fields


def build_game_data(games_df: pd.DataFrame) -> pd.DataFrame:
    combined_tags = []
    for categories, genres, tags in zip(games_df["categories"], games_df["genres"], games_df["tags"]):
        combined_tags.append( # combine categories, genres, and tags into one field
            list(dict.fromkeys( # dedupe while preserving order
                normalize_multi_value_field(categories)
                + normalize_multi_value_field(genres)
                + normalize_multi_value_field(tags)
            ))
        )

    result = pd.DataFrame(
        {
            "item_id": pd.to_numeric(games_df["item_id"], errors="coerce").astype("Int32"),
            "item_name": normalize_text_series(games_df["item_name"]),
            "release_date": parse_date_series(games_df["release_date"]),
            "owners_count": normalize_text_series(games_df["owners_count"]),
            "price": pd.to_numeric(games_df["price"], errors="coerce").astype("Float32").round(2),
            "median_playtime_forever": pd.to_numeric(games_df["median_playtime_forever"], errors="coerce").astype("Float32"),
            "user_reviews": pd.to_numeric(games_df["user_reviews"], errors="coerce").astype("Int32"),
            "developers": games_df["developers"].map(serialize_multi_value_field),
            "publishers": games_df["publishers"].map(serialize_multi_value_field),
            "tags": pd.Series(["|".join(values) if values else "" for values in combined_tags], dtype="string"),
        }
    )

    positive = pd.to_numeric(games_df["positive"], errors="coerce").astype("Float32")
    negative = pd.to_numeric(games_df["negative"], errors="coerce").astype("Float32")
    total_reviews = positive + negative
    positive_ratio = ((positive / total_reviews) * 100).round()
    positive_ratio = positive_ratio.where(total_reviews > 0, 0.0) # if no reviews, set ratio to 0]

    # map ratio and review count to Steam's rating label
    # https://www.reddit.com/r/Steam/comments/ivz45n/what_does_the_steam_ratings_like_very_negative_or/
    result["rating"] = [
                    rating_label(ratio, reviews)
                    for ratio, reviews in zip(positive_ratio, result["user_reviews"])
                ]

    nums = result["owners_count"].str.extract(r'(\d+)\s*-\s*(\d+)').astype(int) # parsing "owners_count" range; "20000 - 50000" -> [20000, 50000]
    result["owners_count"] = ((nums[0] + nums[1]) / 2).astype(int) # take the middle value as int

    result = result[result["item_id"].notna() & result["item_name"].notna()].copy()
    result = result.drop_duplicates(subset=["item_id"], keep="first")
    result = result.sort_values("item_id").reset_index(drop=True)
    return sanitize_df(result)


def build_prod_tables(input_path: Path) -> dict[str, pd.DataFrame]:
    games_df = load_archive_games(input_path)

    game_data_df = build_game_data(games_df)

    # keep only relevant columns for GAME_DATA
    game_data_df = game_data_df[["item_id", "item_name", "release_date", 
                                "owners_count", "price", "median_playtime_forever", "user_reviews", 
                                "rating", "tags", "developers", "publishers"]].copy() # "tags" combines categories, genres, and tags

    return {
        "GAME_DATA": game_data_df,
    }


def main() -> None:
    prod_tables = build_prod_tables(DEFAULT_INPUT_FILE)
    print("Saving production tables to CSV files...")
    for table_name, df in prod_tables.items():
        output_path = DEFAULT_OUTPUT_DIR / OUTPUT_FILENAMES[table_name]
        save_csv(df, output_path)
        print(f"[csv] Wrote {output_path} ({len(df):,} rows)")
if __name__ == "__main__":
    main()
