from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

from helpers import *
from config import PATHS


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / PATHS["prod_ready_dir"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare production-ready CSVs from raw Steam datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where production-ready CSVs will be written.",
    )
    return parser.parse_args()



# parse a date value from the input, return it in "YYYY-MM-DD" format if valid, 
# otherwise return None
def parse_date_value(value: object) -> str | None:
    text = valid_text(value, strip_na_literal=False)
    if text is None or text == "0000-00-00": # treat this as invalid date
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def valid_text(value: object, *, strip_na_literal: bool = True) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in {"\\N", "<NA>"}:
        return None
    if strip_na_literal and text == "NA":
        return None
    return text
# given multiple values for the same field from different sources, choose the first valid one based on the order of the arguments
# also, NaNs are converted into None for consistency in MySQL later
def choose_first_valid(*values: object, strip_na_literal: bool = True) -> str | None:
    for value in values:
        chosen = valid_text(value, strip_na_literal=strip_na_literal)
        if chosen is not None:
            return chosen
    return None


# # given a list of names, build a lookup table with unique names and their corresponding integer IDs
# # this is for the junction table to lookup to
# def build_lookup_table(names: list[str], id_column: str, name_column: str) -> tuple[pd.DataFrame, dict[str, int]]:
#     unique_names = sorted(dict.fromkeys(name for name in names if valid_text(name)))
#     table = pd.DataFrame(
#         {
#             id_column: range(1, len(unique_names) + 1), # auto increment
#             name_column: unique_names,
#         }
#     )
#     return table, dict(zip(table[name_column], table[id_column]))




def read_python_literal_json_lines(file_path: Path) -> pd.DataFrame:
    """
    Read datasets stored as one Python-literal dict per line.
    Several UCSD source files are not strict JSON dumps, they are Python literals with one dict per line. 
    """
    data = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line: # skip empty lines
                data.append(ast.literal_eval(line)) # using ast.literal_eval for safe parsing
    return pd.DataFrame(data)

def prepare_kaggle_tables() -> dict[str, pd.DataFrame]:
    """
    Kaggle Dataset.
    """
    print("")
    print(" ========== Loading and preparing tables... ==========")
    print("")
    games_df = pd.read_csv(BASE_DIR / "tables" / "raw" / "Kaggle" / "games.csv")
    recommendations_df = pd.read_csv(
        BASE_DIR / "tables" / "raw" / "Kaggle" / "recommendations.csv"
    )
    users_df = pd.read_csv(BASE_DIR / "tables" / "raw" / "Kaggle" / "users.csv")
    games_metadata_df = pd.read_json(
        BASE_DIR / "tables" / "raw" / "Kaggle" / "games_metadata.json",
        lines=True,
    )

    games_df = parse_kaggle_games(games_df)
    recommendations_df = parse_kaggle_recommendations(recommendations_df)
    users_df = parse_kaggle_users(users_df)
    games_metadata_df, games_metadata_tags_df = parse_kaggle_games_metadata(
        games_metadata_df
    )

    print("")
    print(f"The columns in the games DataFrame are: {games_df.columns.tolist()}")
    print(f"The columns in the reviews DataFrame are: {recommendations_df.columns.tolist()}")
    print(f"The columns in the users DataFrame are: {users_df.columns.tolist()}")
    print(f"The columns in the games metadata DataFrame are: {games_metadata_df.columns.tolist()}")
    print(f"The columns in the games metadata tags DataFrame are: {games_metadata_tags_df.columns.tolist()}")

    games_df = sanitize_df(games_df)
    recommendations_df = sanitize_df(recommendations_df)
    users_df = sanitize_df(users_df)
    games_metadata_df = sanitize_df(games_metadata_df)
    games_metadata_tags_df = sanitize_df(games_metadata_tags_df)

    return {
        "Kaggle_steam_games.csv": games_df,
        "Kaggle_steam_reviews.csv": recommendations_df,
        "Kaggle_steam_users.csv": users_df,
        "Kaggle_steam_games_metadata.csv": games_metadata_df,
        "Kaggle_steam_games_metadata_tags.csv": games_metadata_tags_df,
    }


def load_prepared_sources() -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    datasets.update(prepare_kaggle_tables())
    return datasets
    


def build_game_data(kaggle_games: pd.DataFrame) -> pd.DataFrame:
    kaggle = (
        kaggle_games[kaggle_games["rating"].between(1, 9, inclusive="both")] # consider only games with valid ratings (which should be the case for all games; this is just for sanity check)
        .sort_values("item_id")
        .drop_duplicates("item_id", keep="first")
        .copy()
    )

    merged = kaggle.copy()
    del kaggle

    result = pd.DataFrame(
        {
            "item_id": merged["item_id"].astype("Int32"),

            "item_name": [
                choose_first_valid(item_name)
                for item_name in merged["title"]
            ],
            "release_date": [
                choose_first_valid(parse_date_value(release_date), strip_na_literal=False)
                for release_date in merged["date_release"]
            ],

            "price_original": merged["price_original"],
            "price_final": merged["price_final"],
            "discount": merged["discount"],


            "rating": merged["rating"].astype("Int8"),
            "positive_ratio": merged["positive_ratio"],
            "user_reviews": merged["user_reviews"],
        }
    )

    result["positive_ratio"] = pd.to_numeric(result["positive_ratio"], errors="coerce").astype("Int8")
    result["user_reviews"] = pd.to_numeric(result["user_reviews"], errors="coerce").astype("Int32")

    result = result[result["item_id"].notna() & result["item_name"].notna()].copy() # exclude rows with no item_id or item_name
    return result.sort_values("item_id").reset_index(drop=True)


def build_prod_tables() -> dict[str, pd.DataFrame]:
    
    datasets = load_prepared_sources() # run for around 1.2 mins

    kaggle_games = datasets["Kaggle_steam_games.csv"].copy()
    kaggle_reviews = datasets["Kaggle_steam_reviews.csv"].copy()
    kaggle_users = datasets["Kaggle_steam_users.csv"].copy()
    kaggle_meta_tags = datasets["Kaggle_steam_games_metadata_tags.csv"].copy()

    del datasets # free memory

    print("")
    print("Building production tables...")

    game_data = build_game_data(kaggle_games)
    valid_item_ids = set(game_data["item_id"].astype("Int32").tolist()) # set uniqueness

    tag_source = kaggle_meta_tags.copy()
    del kaggle_meta_tags # free memory

    tag_source["tag"] = tag_source["tag"].map(valid_text) # sanity check to ensure tags are valid text values
    tag_source = tag_source[tag_source["item_id"].isin(valid_item_ids) & tag_source["tag"].notna()]
    tag_source = tag_source.drop_duplicates(subset=["item_id", "tag"], keep="first").copy()

    # len(kaggle_users) - len(kaggle_users.drop_duplicates(subset=["user_id"], keep="first")) # returns 0, confirmed no duplicates
    user_kaggle_df = kaggle_users.copy()
    del kaggle_users # free memory

    user_kaggle_df["steam_id"] = user_kaggle_df["steam_id"].astype("Int32")
    user_kaggle_df["products"] = pd.to_numeric(user_kaggle_df["products"], errors="coerce").astype("Int32")
    user_kaggle_df["reviews"] = pd.to_numeric(user_kaggle_df["reviews"], errors="coerce").astype("Int32")
    user_kaggle_df = user_kaggle_df.sort_values("steam_id").reset_index(drop=True) # sort by steam_id
    valid_user_kaggle_ids = set(user_kaggle_df["steam_id"].astype("Int32").tolist()) # set uniqueness

    game_review_kaggle_df = kaggle_reviews[ # filter reviews to only include those with valid item_ids and user_ids
        kaggle_reviews["item_id"].isin(valid_item_ids)
        & kaggle_reviews["steam_id"].isin(valid_user_kaggle_ids)
    ][["review_id", "item_id", "steam_id", "hours", "date", "recommend", "helpful"]].copy()
    del kaggle_reviews # free memory

    game_review_kaggle_df["review_id"] = pd.to_numeric(game_review_kaggle_df["review_id"], errors="coerce").astype("Int32")


    # ...
    # len(game_review_kaggle_df["date"][
    #     game_review_kaggle_df["date"].notna() 
    #     & (game_review_kaggle_df["date"] != "\\N") 
    #     & (game_review_kaggle_df["date"] != "") 
    #     & (game_review_kaggle_df["date"] != "<NA>")
    #     & (game_review_kaggle_df["date"] != "0000-00-00")
    #     & (game_review_kaggle_df["date"] != pd.NaT)
    #     & (game_review_kaggle_df["date"] != "NA")
    # ]
    # ) - len(game_review_kaggle_df["date"])
    # don't need parse_date_value, confirmed no invalid data values in "date"
    # game_review_kaggle_df["date"] = game_review_kaggle_df["date"].map(parse_date_value) # parse date values and convert invalid ones to None
    game_review_kaggle_df["date"] = pd.to_datetime(game_review_kaggle_df["date"], errors="coerce")
    game_review_kaggle_df = game_review_kaggle_df.sort_values("date", ascending=False)

    # drop duplicate reviews from the same user for the same game, keeping only the most recent one based on date  
    dup_row_count = game_review_kaggle_df.duplicated(subset=["steam_id", "item_id"]).sum()
    game_review_kaggle_df = game_review_kaggle_df.drop_duplicates(subset=["steam_id", "item_id"], keep="first") 
    print("")
    print(f" Dropped {dup_row_count:,} duplicate rows in game review table")
    print(f" Final count: {len(game_review_kaggle_df):,} raw rows in game review table")
    print("")

    # re-sort based on review_id
    game_review_kaggle_df = game_review_kaggle_df.sort_values("review_id").reset_index(drop=True)

    print("Done building production tables.")
    print("")

    return {
        "GAME_USER.csv": user_kaggle_df,
        "GAME_REVIEW.csv": game_review_kaggle_df,
        "GAME_DATA.csv": game_data,
        "GAME_TAG.csv": tag_source,
    }


def main() -> None:
    args = parse_args()
    prod_tables = build_prod_tables()
    print("Saving production tables to CSV files...")
    for filename, df in prod_tables.items():
        output_path = args.output_dir / filename
        save_csv(df, output_path)
        print(f"[csv] Wrote {output_path} ({len(df):,} rows)")


    # run
    # robocopy "C:\Users\salir\OneDrive\Documents\Personal Project\Steam-Recommendation-System\tables\production" "C:\ProgramData\MySQL\MySQL Server 8.0\Uploads" /E /Z /XA:H /W:5 /R:5

if __name__ == "__main__":
    main()
