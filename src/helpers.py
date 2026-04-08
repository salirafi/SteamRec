import ast
import re
import pandas as pd
from pathlib import Path
import numpy as np
import csv
from typing import Any
import json
from dataclasses import dataclass

from scipy import sparse
from sqlalchemy import text

try:
    from src.config import QUERY_CONFIG, get_factor_column_names
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution inside src/
    from config import QUERY_CONFIG, get_factor_column_names

INPUT_DIR  = Path("./tables")
# OUTPUT_DIR = Path("./tables/staging")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT_MAP = {
    "Overwhelmingly Negative": 1,
    "Very Negative":           2,
    "Negative":                3,
    "Mostly Negative":         4,
    "Mixed":                   5,
    "Mostly Positive":         6,
    "Positive":                7,
    "Very Positive":           8,
    "Overwhelmingly Positive": 9,
}

BOOLEAN_MAP = {
    "True": 1,
    True: 1,
    1: 1,
    "False": 0,
    False: 0,
    0: 0,
}

# ==================================
# HELPERS FOR PRE_PROCESSING STAGES
# ==================================

# explicitly shows the state for easier MySQL import
def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_path,
        index=False,
        na_rep="", # use empty string for NaN values
        lineterminator="\r\n", # use Windows-style line endings for better compatibility with MySQL LOAD DATA INFILE
        sep=",",
        encoding="utf-8",
        quoting=csv.QUOTE_ALL, # quote all fields
    )

def sanitize_text(val):
    """
    For safe CSV export for MySQL.
    """
    if not isinstance(val, str):
        return val
    val = val.replace('\\', '/') # replace backslash with forward slash
    val = val.replace('\x00', '') # strip null bytes
    val = val.replace('\r', ' ') # strip carriage returns within fields
    return val


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].apply(sanitize_text)
    return df


def to_nullable_int(series: pd.Series, dtype: str) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(dtype)


def to_nullable_float(series: pd.Series, dtype: str = "Float32") -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(dtype)


# Helpers for UCSD dataset
# ==============================


def parse_col(val):
    """
    Safely parse a value that may be a Python list/dict literal stored as a string.
    Returns the original value if it's already a list/dict, or if parsing fails.
    """
    if isinstance(val, (list, dict)): # this includes empty lists/dicts
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return None
    return None

# def parse_is_compensated(val):
#     """
#     Non-empty, non-NaN string -> True
#     Empty string or NaN       -> False
#     """
#     if not isinstance(val, str):
#         return False
#     return val.strip() != ""


def parse_posted_date(posted_str):
    """
    Convert 'Posted November 5, 2011.' -> '2011-11-05'
    Returns None if the string can't be parsed.
    """
    if not isinstance(posted_str, str) or not posted_str.strip():
        return None
    # strip the 'Posted ' prefix and trailing punctuation; case insensitive
    cleaned = re.sub(r"^Posted\s+", "", posted_str.strip(), flags=re.IGNORECASE).rstrip(".")
    try:
        return pd.to_datetime(cleaned, format="%B %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        try:
            # fallback -> handle edge cases like "Posted 5 Nov, 2011" or "Posted Nov 5 2011"
            return pd.to_datetime(cleaned).strftime("%Y-%m-%d")
        except Exception:
            return None
        

def parse_last_edited_date(edited_str):
    """
    Convert 'Last edited November 5, 2011.' -> '2011-11-05'
    Returns None if the string can't be parsed.
    """
    if not isinstance(edited_str, str) or not edited_str.strip():
        return None
    # strip the 'Last edited ' prefix and trailing punctuation; case insensitive
    cleaned = re.sub(r"^Last edited\s+", "", edited_str.strip(), flags=re.IGNORECASE).rstrip(".")
    try:
        return pd.to_datetime(cleaned, format="%B %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        try:
            # fallback -> handle edge cases like "Last edited 5 Nov, 2011" or "Last edited Nov 5 2011"
            return pd.to_datetime(cleaned).strftime("%Y-%m-%d")
        except Exception:
            return None


def parse_helpful_count(helpful_str):
    """
    'x out of y people (m%) found this review helpful' -> x
    'No ratings yet' -> 0
    """
    if not isinstance(helpful_str, str) or not helpful_str.strip():
        return 0 # treat empty or non-string helpful as 0 helpful votes
    match = re.match(r"^(\d+)", helpful_str.strip())
    return int(match.group(1)) if match else 0 # if no helpful count found, return 0


def parse_helpful_pct(helpful_str):
    """
    'x out of y people (m%) found this review helpful' -> m (as float)
    'No ratings yet' -> None
    """
    if not isinstance(helpful_str, str) or not helpful_str.strip():
        return None # treat empty or non-string helpful as None
    match = re.search(r"\((\d+(?:\.\d+)?)%\)", helpful_str)

    
    return float(match.group(1)) if match else 0.0 # if no percentage found, return 0.0


def strip_dollar(price_str):
    """
    '$8.99' -> float(8.99)
    Returns None if the value can't be converted.
    """
    if not isinstance(price_str, str):
        return None
    cleaned = price_str.replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None
    
def strip_pct(price_str):
    """
    '25%' -> float(25.0)
    Returns None if the value can't be converted.
    """
    if not isinstance(price_str, str):
        return None
    cleaned = price_str.replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None
    

def parse_price(price_str):
    """
      - Contains 'free' (case-insensitive) -> (None, True)
      - Parses cleanly as float -> (float, False)
      - Anything else (unparseable) -> (None, True)
    """
    if (isinstance(price_str, float) and np.isnan(price_str)) or (isinstance(price_str, str) and price_str == ""): # can be float or "Free to Play"
        # already null/NaN coming in
        return None, None
 
    if isinstance(price_str, str) and re.search(r"free", price_str, re.IGNORECASE):
        return None, True
 
    try:
        if isinstance(price_str, float):
            return price_str, False
        else:
            return None, None
    except ValueError:
        return None, True
 

def encode_sentiment(val):
    """
    Maps sentiment string to ordered int 1-9.
    NaN / unrecognised values -> None.
    """
    if not isinstance(val, str):
        return None
    return SENTIMENT_MAP.get(val.strip(), None)
 
 
def explode_tag_column(df, id_col, tag_col, tag_type):
    """
    Given a DataFrame with a list-of-strings column, explode it into (id_col, tag, tag_type) rows.
    """
    subset = df[[id_col, tag_col]].copy()
    subset = subset.explode(tag_col)
    subset[tag_col] = subset[tag_col].str.strip()
    subset = subset[subset[tag_col].str.len() > 0] # drop empty tags
    subset = subset.rename(columns={tag_col: "tag"})
    subset["tag_type"] = tag_type
    subset = subset.rename(columns={id_col: "item_id"})
    return subset.reset_index(drop=True)


# Main helper functions for UCSD dataset
# ==============================


def explode_bundles(df: pd.DataFrame):
    """
    Returns two DataFrames:
      bundles_df -- one row per bundle (bundle-level metadata)
      bundle_items_df -- one row per (bundle, game)
    """
    
    # bundle_price: '$25' -> 25.0
    for col in ["bundle_price", "bundle_final_price"]:
        df[col] = to_nullable_float(df[col].apply(strip_dollar))
 
    # bundle_discount: '25%' -> 25.0
    df["bundle_discount"] = to_nullable_int(
        df["bundle_discount"].apply(strip_pct).round(),
        "Int8",
    )
    df["bundle_id"] = to_nullable_int(df["bundle_id"], "Int32")
 
    # bundle-level staging (no items column)
    bundles_df = df.drop(columns=["items"]).copy()
 
    # explode items column into bundle_items_df
    df["items"] = df["items"].apply(parse_col)
    bad = df["items"].isna().sum()
    if bad:
        print(f"{bad} bundles had unparseable 'items'; dropped.")
    df = df.dropna(subset=["items"])
 
    exploded = df[["bundle_id", "items"]].explode("items").reset_index(drop=True)
    items_expanded = pd.json_normalize(exploded["items"])
    bundle_items_df = pd.concat(
        [exploded["bundle_id"].reset_index(drop=True), items_expanded],
        axis=1
    )
 
    # clean item-level columns
 
    bundle_items_df["discounted_price"] = to_nullable_float(
        bundle_items_df["discounted_price"].apply(strip_dollar)
    ) # "discounted_price" is the key in the items dict for the price after discount
    
    bundle_items_df["item_id"] = to_nullable_int( # "item_id" is the key in the items dict for the app id
        bundle_items_df["item_id"], "Int32"
    )

    # drop rows with no valid item_id
    bad_ids = bundle_items_df["item_id"].isna().sum()
    if bad_ids:
        print(f"{bad_ids} bundle item rows had null item_id; dropped.")
    bundle_items_df = bundle_items_df.dropna(subset=["item_id"])
 
    # keep staging columns
    # these are the keys in the items dict that we care about for staging
    # I drop item_url, item_name, and genre because we can get them from game details table via item_id
    # discounted_price is kept since items may be discounted differently within bundles than their standalone price in the game details table
    item_keep = ["bundle_id", "item_id", "discounted_price"]
    bundle_items_df = bundle_items_df[[c for c in item_keep if c in bundle_items_df.columns]]
 
    print(f"Exploding successful: {len(bundles_df):,} bundles, {len(bundle_items_df):,} bundle-item rows.")
    return bundles_df, bundle_items_df


def explode_user_items(df: pd.DataFrame) -> pd.DataFrame:

    print(f"There are {len(df)} rows before explode.")

    # parse the items column in case it was stored as a string
    df["items"] = df["items"].apply(parse_col)

    empty_lists = (df['items'].apply(len) == 0).sum()
    print(f"{empty_lists} rows where 'items' is an empty list (the user has no items); dropped.")

    # remove rows where 'items' is an empty list, since they won't contribute any rows after explode anyway
    df = df[df['items'].apply(len) > 0].reset_index(drop=True)

    # drop rows where parsing failed
    bad = df["items"].isna().sum()
    if bad:
        print(f"{bad} rows had unparseable 'items'; dropped.")
    df = df.dropna(subset=["items"])

    # explode: one row per (user, item)
    df = df.explode("items").reset_index(drop=True)

    # unpack the dict into columns
    items_expanded = pd.json_normalize(df["items"])
    df = df.drop(columns=["items"]).reset_index(drop=True)
    df = pd.concat([df, items_expanded], axis=1)

    # the columns are based on the keys in the items dict
    df["item_id"] = to_nullable_int(df["item_id"], "Int32")
    df["items_count"] = to_nullable_int(df["items_count"], "Int32")
    df["playtime_forever"] = to_nullable_float(
        pd.to_numeric(df["playtime_forever"], errors="coerce").fillna(0)
    ) # NaN playtime -> 0.0
    df["playtime_2weeks"] = to_nullable_float(
        pd.to_numeric(df["playtime_2weeks"], errors="coerce").fillna(0)
    ) # NaN playtime -> 0.0

    # drop rows with no valid item_id
    bad_ids = df["item_id"].isna().sum()
    if bad_ids:
        print(f"{bad_ids} rows had null item_id after cast; dropped.")
    df = df.dropna(subset=["item_id"])

    df["steam_id"] = to_nullable_int(df["steam_id"], "Int64")

    # keep only the columns needed for staging
    keep = ["user_id", "steam_id", "items_count", "item_id",
            "playtime_forever", "playtime_2weeks"]
    df = df[[c for c in keep if c in df.columns]]

    print(f"There are {len(df):,} rows after explode.")

    return df


def explode_user_reviews(df: pd.DataFrame) -> pd.DataFrame:
    
    print(f"There are {len(df)} rows before explode.")

    df["reviews"] = df["reviews"].apply(parse_col)

    empty_lists = (df['reviews'].apply(len) == 0).sum()
    print(f"{empty_lists} rows where 'reviews' is an empty list (the user has no reviews); dropped.")

    # remove rows where 'reviews' is an empty list, since they won't contribute any rows after explode anyway
    df = df[df['reviews'].apply(len) > 0].reset_index(drop=True)
 
    bad = df["reviews"].isna().sum()
    if bad:
        print(f"{bad} rows had unparseable 'reviews'; dropped.")
    df = df.dropna(subset=["reviews"])
 
    # explode: one row per (user, review)
    df = df.explode("reviews").reset_index(drop=True)
 
    # unpack the dict
    reviews_expanded = pd.json_normalize(df["reviews"])
    df = df.drop(columns=["reviews"]).reset_index(drop=True)
    df = pd.concat([df, reviews_expanded], axis=1)
 
 
    # posted date
    df["posted"] = df["posted"].apply(parse_posted_date) # this will return None for unparseable or missing dates
    df = df.rename(columns={"posted": "date_posted"})
 
    # helpful: 'No ratings yet' / 'x out of y people ...' -> int and float
    raw_helpful       = df["helpful"].copy()
    df["helpful"] = to_nullable_int(
        raw_helpful.apply(parse_helpful_count),
        "Int32",
    ) # this will return 0 for unparseable and missing helpful strings
    df["helpful_pct"] = to_nullable_float(
        raw_helpful.apply(parse_helpful_pct)
    ) # this will return 0.0 for unparseable and missing helpful strings
 
    # funny: same as helpful but only for parse_helpful_count
    raw_funny       = df["funny"].copy()
    df["funny"] = to_nullable_int(raw_funny.apply(parse_helpful_count), "Int32")
 
    # last_edited
    df["last_edited"] = df["last_edited"].apply(parse_last_edited_date) # this will return None for unparseable or missing dates
    df = df.rename(columns={"last_edited": "date_last_edited"})
 
    # item_id to int
    df["item_id"] = to_nullable_int(df["item_id"], "Int32")
 
    # recommend: change to binary 1/0 for MySQL
    df["recommend"] = to_nullable_int(df["recommend"].map(BOOLEAN_MAP), "Int8")
 
    # review text: replace empty string with None
    df["review"] = df["review"].replace("", None)
 
    # drop rows with no valid item_id
    bad_ids = df["item_id"].isna().sum()
    if bad_ids:
        print(f"{bad_ids} rows had null item_id after cast; dropped.")
    df = df.dropna(subset=["item_id"])
 
    # keep staging columns
    keep = ["user_id", "user_url", "item_id", "recommend", "review",
            "date_posted", "date_last_edited", "helpful", "helpful_pct", "funny"]
    df = df[[c for c in keep if c in df.columns]]
 
    # add source tag
    df["source"] = "australian"
 
    print(f"There are {len(df):,} rows after explode.")
    return df


def parse_steam_games(df: pd.DataFrame):

    """
    Returns:
      games_df: cleaned game-level rows
      tags_df: exploded (app_id, tag, tag_type) for genres + tags
      specs_df: exploded (app_id, spec)
    """
 
    # id -> item_id, cast to Int32
    df["id"] = to_nullable_int(df["id"], "Int32")
    bad_ids = df["id"].isna().sum()
    if bad_ids:
        print(f"{bad_ids} rows had unparseable id; dropped.")
    df = df.dropna(subset=["id"])
    df = df.rename(columns={"id": "item_id"})

    # early_access: change to binary 1/0 for MySQL
    df["early_access"] = to_nullable_int(df["early_access"].map(BOOLEAN_MAP), "Int8")
 
    # price: parse price and also create is_free boolean
    parsed_prices = df["price"].apply(parse_price)
    df["price"] = to_nullable_float(parsed_prices.apply(lambda x: x[0]))
    df["is_free"] = to_nullable_int(
        parsed_prices.apply(lambda x: x[1]).map(BOOLEAN_MAP),
        "Int8",
    )
 
    # discount_price: parse price, keep as float (NaNs will be dropped in MySQL import since column is not nullable)
    df["discount_price"] = to_nullable_float(df["discount_price"])
 
    # metascore; already in int, just coerce into NaNs
    df["metascore"] = to_nullable_int(df["metascore"], "Int32")
 
    # sentiment ordered int 1-9, with NaN for unrecognised or missing sentiment
    df["sentiment"] = to_nullable_int(df["sentiment"].apply(encode_sentiment), "Int8")
 
    # explode genres -> tag_type = 'genre'
    genres_df = explode_tag_column(df, "item_id", "genres", "genre")
 
    # explode tags -> tag_type = 'tag'
    tags_df = explode_tag_column(df, "item_id", "tags", "tag")
 
    # combine genres + tags into one unified tag table
    all_tags_df = pd.concat([genres_df, tags_df], ignore_index=True)
 
    # explode specs -> tag_type = 'spec'
    specs_df = explode_tag_column(df, "item_id", "specs", "spec")
    specs_df = specs_df.drop(columns=["tag_type"]).rename(columns={"tag": "spec"})
 
    # drop the now-exploded list columns from game-level df
    df = df.drop(columns=["genres", "tags", "specs"])

    df = df.rename(columns={"app_name": "item_name"})
 
    print(f"There are {len(df):,} games, "
          f"{len(all_tags_df):,} tag+genre rows, "
          f"{len(specs_df):,} spec rows.")
    return df, all_tags_df, specs_df


def parse_bundle_item_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits the genre string in stg_bundle_items into one row per genre,
    matching the same (app_id, tag, tag_type) format as stg_steam_game_tags
    so they can be unioned later.
 
    Input:  stg_bundle_items.csv  (from explode_nested.py)
    Output: stg_bundle_item_genres.csv
    """
 
    # genre is a comma-separated string: "Adventure, Indie, RPG"
    df["genre"] = df["genre"].fillna("")
    df["genre"] = df["genre"].str.split(",")
    df = df.explode("genre")
    df["genre"] = df["genre"].str.strip()
    df = df[df["genre"].str.len() > 0]
    df["item_id"] = to_nullable_int(df["item_id"], "Int32")
 
    result = df[["item_id", "genre"]].rename(
        columns={"item_id": "item_id", "genre": "tag"}
    ).copy()
    result["tag_type"] = "genre"
    result = result.reset_index(drop=True)
 
    print(f"  [ok] {len(result):,} bundle item genre rows.")
    return result


def parse_steam_reviews(df: pd.DataFrame) -> pd.DataFrame:

    print("There are {} rows in the steam reviews DataFrame.".format(df.shape[0]))
 
    df = df.drop(columns=["page", "page_order"], errors="ignore") # drop columns with no clear meaning for the recommender
 
    # product_id -> item_id, cast to Int32
    df = df.rename(columns={"product_id": "item_id"})
    df["item_id"] = to_nullable_int(df["item_id"], "Int32")
    bad_items = df["item_id"].isna().sum()
    if bad_items:
        print(f"{bad_items} rows had unparseable item_id; dropped.")
    df = df.dropna(subset=["item_id"])
 
    # user_id -> steam_id, cast to Int64
    df = df.rename(columns={"user_id": "steam_id"})
    df["steam_id"] = to_nullable_int(df["steam_id"], "Int64")
    null_steam = df["steam_id"].isna().sum()
    if null_steam:
        print(f"{null_steam} rows have null steam_id; kept as pd.NA in Int64.")

    # early_access: change to binary 1/0 for MySQL
    df["early_access"] = to_nullable_int(df["early_access"].map(BOOLEAN_MAP), "Int8")
 
    # found_funny -> funny
    df = df.rename(columns={"found_funny": "funny"})
    df["funny"] = to_nullable_int(
        pd.to_numeric(df["funny"], errors="coerce").fillna(0),
        "Int32",
    )
 
    df["products"] = to_nullable_int(
        pd.to_numeric(df["products"], errors="coerce").fillna(0),
        "Int32",
    )
    df["hours"] = to_nullable_float(df["hours"])
 
    # compensation -> is_compensated boolean
    # assume that every kind of non-empty values in "compensation" is a form of compensation
    df["is_compensated"] = df["compensation"].apply(lambda val: isinstance(val, str))
    df = df.drop(columns=["compensation"])

    # same for is_compensated
    df["is_compensated"] = to_nullable_int(df["is_compensated"].map(BOOLEAN_MAP), "Int8")
 
    # add source tag
    df["source"] = "ucsd"
 
    print(f"There are {len(df):,} rows after cleaning.")
    return df


# Helpers for Kaggle dataset
# ==============================


def explode_tag_column(df, id_col, tag_col):
    """
    Explodes a list-of-strings column into (id_col, tag) rows.

    """
    subset = df[[id_col, tag_col]].copy()
    subset = subset.explode(tag_col)
    subset[tag_col] = subset[tag_col].str.strip()
    subset = subset[subset[tag_col].str.len() > 0] # drop empty tags
    subset = subset.rename(columns={tag_col: "tag", id_col: "item_id"})
    # subset["tag_type"] = tag_type
    return subset.reset_index(drop=True)


def parse_kaggle_games(df: pd.DataFrame) -> pd.DataFrame:
 
    # app_id -> item_id
    df = df.rename(columns={"app_id": "item_id"})

    df["item_id"] = to_nullable_int(df["item_id"], "Int32")
    df["positive_ratio"] = to_nullable_int(df["positive_ratio"], "Int8")
    df["user_reviews"] = to_nullable_int(df["user_reviews"], "Int32")
 
    # rating -> ordered int 1-9
    df["rating"] = to_nullable_int(df["rating"].map(SENTIMENT_MAP), "Int8")
    unmapped = df["rating"].isna().sum()
    if unmapped:
        print(f"{unmapped} rows had unrecognised or null rating; set to NA.")
    
    # change boolean to binary 1/0 for MySQL
    df["win"] = to_nullable_int(df["win"].map(BOOLEAN_MAP), "Int8")
    df["mac"] = to_nullable_int(df["mac"].map(BOOLEAN_MAP), "Int8")
    df["linux"] = to_nullable_int(df["linux"].map(BOOLEAN_MAP), "Int8")
    df["steam_deck"] = to_nullable_int(df["steam_deck"].map(BOOLEAN_MAP), "Int8")

    df["price_final"] = to_nullable_float(df["price_final"])
    df["price_original"] = to_nullable_float(df["price_original"])
    df["discount"] = to_nullable_int(df["discount"].round(), "Int8")
 
    print(f"There are {len(df):,} games.")
    return df


def parse_kaggle_recommendations(df: pd.DataFrame) -> pd.DataFrame:

    # app_id -> item_id
    df = df.rename(columns={
        "app_id":       "item_id",
        "user_id":      "steam_id",
        "is_recommended": "recommend",
    })

    df["item_id"] = to_nullable_int(df["item_id"], "Int32")
    df["steam_id"] = to_nullable_int(df["steam_id"], "Int32")
    df["helpful"] = to_nullable_int(df["helpful"], "Int32")
    df["funny"] = to_nullable_int(df["funny"], "Int32")
    df["review_id"] = to_nullable_int(df["review_id"], "Int32")
    df["hours"] = to_nullable_float(df["hours"])

    # recommend: change to binary 1/0 for MySQL
    df["recommend"] = to_nullable_int(df["recommend"].map(BOOLEAN_MAP), "Int8")
 
 
    print(f"There are {len(df):,} reviews.")
    return df


def parse_kaggle_users(df: pd.DataFrame) -> pd.DataFrame:

    # user_id -> steam_id
    df = df.rename(columns={"user_id": "steam_id"})

    df["steam_id"] = to_nullable_int(df["steam_id"], "Int32")
    df["products"] = to_nullable_int(df["products"], "Int32")
    df["reviews"] = to_nullable_int(df["reviews"], "Int32")
 
 
    print(f"There are {len(df):,} users.")
    return df

 
def parse_kaggle_games_metadata(df: pd.DataFrame):
    """
    Returns:
      meta_df; (item_id, description, source)
      tags_df; (item_id, tag)
    """
    # app_id -> item_id
    df = df.rename(columns={"app_id": "item_id"})
    df["item_id"] = to_nullable_int(df["item_id"], "Int32")
 
    # description: empty string -> None
    df["description"] = df["description"].replace("", None)
 
    # explode tags
    tags_df = explode_tag_column(df, "item_id", "tags") # only tags, no genres
 
    # drop tags column from metadata df
    meta_df = df.drop(columns=["tags"])
 
    print(f"There are {len(meta_df):,} metadata rows, {len(tags_df):,} tag rows.")
    return meta_df, tags_df


# ==================================
# HELPERS FOR RECOMMENDER ANALYSIS
# ==================================


@dataclass
class TrainingArtifacts:
    interactions_df: pd.DataFrame
    items_df: pd.DataFrame
    item_matrix: sparse.csr_matrix
    item_id_to_idx: dict[int, int]
    item_rerank_df: pd.DataFrame | None = None


@dataclass
class RuntimeArtifacts:
    items_df: pd.DataFrame
    item_matrix: sparse.csr_matrix
    item_id_to_idx: dict[int, int]
    item_rerank_df: pd.DataFrame | None = None


@dataclass
class ALSArtifacts:
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_id_to_idx_als: dict[int, int]
    item_id_to_idx_als: dict[int, int]

# convert multi-valie fields into lists
def to_list(x: object) -> list[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []  # empty list for NaNs
    if isinstance(x, list):
        values = x
    else:  # string with values separated by "|"
        values = [v.strip() for v in str(x).split("|") if v.strip()]

    if values == [r"\N"]:
        return []  # empty list for literal "\N"; sanity check
    return list(dict.fromkeys(values))


def _safe_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    smin = s.min()
    smax = s.max()
    if smax <= smin:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - smin) / (smax - smin)
