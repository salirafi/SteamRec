from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1" # limit to single thread

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sqlalchemy import create_engine, text

try:
    from src.config import PATHS, QUERY_CONFIG, get_db_url, get_recommender_config
    from src.helpers import _safe_minmax, save_csv, to_list
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution inside src/
    from config import PATHS, QUERY_CONFIG, get_db_url, get_recommender_config
    from helpers import _safe_minmax, save_csv, to_list


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / PATHS["rec_matrices_dir"]
RUNTIME_OUTPUT_DIR = BASE_DIR / PATHS["output_dir"]

config = get_recommender_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare recommender-ready CSVs for bulk LOAD DATA ingestion."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where recommender CSVs will be written.",
    )
    parser.add_argument(
        "--item-similarity-top-k",
        type=int,
        default=150,
        help="How many similar items to export for each source item.",
    )
    parser.add_argument(
        "--runtime-output-dir",
        type=Path,
        default=RUNTIME_OUTPUT_DIR,
        help="Directory where runtime item artifacts for the Flask recommender will be written.",
    )
    return parser.parse_args()


# save latent factors to CSV with columns:
    # id_column: original steam_id or item_id
    # idx_column: corresponding row index in the factor matrix
    # factors: JSON-encoded list of latent factor values for that user/item
def save_factor_csv(
    output_path: Path,
    id_to_idx: dict[int, int],
    factors,
    id_column: str,
    idx_column: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        writer.writerow([id_column, idx_column, "factors"])
        for entity_id, factor_idx in sorted(id_to_idx.items(), key=lambda item: item[1]):
            writer.writerow(
                [
                    int(entity_id),  # item/user id
                    int(factor_idx),  # latent factor index
                    json.dumps([float(value) for value in factors[factor_idx]]),  # factor values
                ]
            )


def save_index_csv(
    output_path: Path,
    id_to_idx: dict[int, int],
    id_column: str,
    idx_column: str,
) -> None:
    df = pd.DataFrame(
        [
            {id_column: int(entity_id), idx_column: int(factor_idx)}
            for entity_id, factor_idx in sorted(id_to_idx.items(), key=lambda item: item[1])
        ]
    )
    save_csv(df, output_path)


def save_item_similarity_csv(
    output_path: Path,
    items_df: pd.DataFrame,
    item_matrix,
    top_k: int,
) -> None:
    """
    Compute cosine similarity between items and save top-k similar items for each item to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    item_ids = items_df["item_id"].astype(int).tolist()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        writer.writerow(["source_item_id", "similar_item_id", "similarity_rank", "similarity_score"])

        # loop over each item and compute similarities to all other items
        for item_idx, source_item_id in enumerate(item_ids):
            sims = cosine_similarity(item_matrix[item_idx], item_matrix).ravel()
            sims[item_idx] = -1.0  # exclude self-similarity by setting it negative
            actual_k = min(top_k, max(len(sims) - 1, 0))
            if actual_k <= 0:
                continue

            top_idx = sims.argpartition(-actual_k)[-actual_k:]  # get indices of top-k similar items (unsorted)
            top_idx = top_idx[sims[top_idx].argsort()[::-1]]  # sort top-k indices by similarity score

            # loop over top-k similar items and write to CSV
            for rank, similar_idx in enumerate(top_idx, start=1):
                writer.writerow(
                    [
                        int(source_item_id),
                        int(item_ids[int(similar_idx)]),
                        rank,
                        float(sims[int(similar_idx)]),
                    ]
                )


def save_runtime_artifacts(
    output_dir: Path,
    items_df: pd.DataFrame,
    # item_rerank_df: pd.DataFrame,
    item_factors: np.ndarray,
    item_id_to_idx_als: dict[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    items_runtime_df = items_df.copy()
    items_runtime_df["item_id"] = pd.to_numeric(items_runtime_df["item_id"], errors="coerce").astype("int64")
    items_runtime_df.to_parquet(output_dir / "items_used.parquet", index=False)

    # item_rerank_runtime_df = item_rerank_df.copy()
    # item_rerank_runtime_df["item_id"] = pd.to_numeric(item_rerank_runtime_df["item_id"], errors="coerce").astype("int64")
    # item_rerank_runtime_df.to_parquet(output_dir / "item_rerank.parquet", index=False)

    np.save(output_dir / "item_factors.npy", np.asarray(item_factors, dtype=np.float32))

    with (output_dir / "item_id_to_idx_als.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {str(int(item_id)): int(idx) for item_id, idx in sorted(item_id_to_idx_als.items(), key=lambda item: item[1])},
            handle,
            ensure_ascii=True,
            indent=2,
        )

    metadata = {
        "n_items_used": int(len(items_runtime_df)),
        "n_item_factors": int(np.asarray(item_factors).shape[0]),
        "n_latent_factors": int(np.asarray(item_factors).shape[1]) if np.asarray(item_factors).ndim == 2 else 0,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)


# MySQL query to get item data
def load_items(engine) -> pd.DataFrame:
    items_df = pd.read_sql(text(QUERY_CONFIG["item_query"]), engine)
    print(f"Loaded {len(items_df):,} items.")
    return items_df


# My SQL query to get user-item interactions
# also compute the interaction score with log(hours) * recommend
def load_interactions(engine) -> pd.DataFrame:
    interactions_df = pd.read_sql(text(QUERY_CONFIG["interaction_query"]), engine)
    print(f" Loaded {len(interactions_df):,} raw rows")

    interactions_df["hours"] = pd.to_numeric(interactions_df["hours"], errors="coerce").fillna(0.0)
    interactions_df["recommend"] = pd.to_numeric(interactions_df["recommend"], errors="coerce").fillna(0)
    interactions_df["interaction_score"] = (  # calculate interaction score based on recommend and hours
        np.log1p(interactions_df["hours"].fillna(0.0))  # apply log transformation to hours to reduce skewness
        * interactions_df["recommend"].map({1: 1.5, 0: 0.5}).fillna(0.0)  # give more weight to recommend=1 and less weight to recommend=0
    ).astype(np.float32)
    interactions_df = interactions_df[interactions_df["interaction_score"] > 0].copy()  # filter out meaningless interactions
    return interactions_df


def filter_interactions_to_runtime_items(
    interactions_df: pd.DataFrame,
    items_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only interactions whose items exist in the runtime item catalog.

    This keeps ALS item factors aligned with the metadata used later by the Flask app.
    Without this filter, CF can recommend items that exist in the interaction matrix
    but were removed from items_df (no tags).
    """
    valid_item_ids = set(pd.to_numeric(items_df["item_id"], errors="coerce").dropna().astype(int).tolist())
    filtered = interactions_df[interactions_df["item_id"].isin(valid_item_ids)].copy()
    n_removed = len(interactions_df) - len(filtered)
    print(f"Filtered out {n_removed:,} interactions whose item_id is missing from runtime items.")
    return filtered


# remove items with no tags (tags are the backbone of the content-based similarity)
def filter_sparse_items(
    items_df: pd.DataFrame,
    min_tags: int = 1,
) -> pd.DataFrame:
    df = items_df.copy()
    df["_tags"] = df["tags"].apply(to_list)

    has_tags = df["_tags"].apply(len) >= min_tags
    mask = has_tags  # keep if it has tags

    n_removed = (~mask).sum()
    print(f"Filtered out {n_removed:,} items with no tags ({n_removed / len(df) * 100:.1f}% of catalog)")

    return items_df[mask].reset_index(drop=True)


# build item feature matrix for content-based similarity
def fit_items(items_df: pd.DataFrame) -> tuple[pd.DataFrame, sparse.csr_matrix, dict[int, int]]:
    df = items_df.copy()
    df["tags"] = df["tags"].apply(to_list)

    dates = pd.to_datetime(df["release_date"], errors="coerce")  # coerce to pd.NaT
    current_year = pd.Timestamp.today().year

    # fill missing years with 2003 (year Steam founded)
    # adjust if the user wants to treat them differently
    df["release_year"] = dates.dt.year.fillna(2003).astype(int)
    df["game_age_inv"] = 1 / (current_year - df["release_year"] + 1)  # take inverse of age to give more weight to newer games

    # convert sparse categorical features to binary indicator matrices
    tags = MultiLabelBinarizer(sparse_output=True).fit_transform(df["tags"])

    # age_inv = df[["game_age_inv"]].copy()
    # age_inv = pd.DataFrame(
    #     MaxAbsScaler().fit_transform(age_inv), # scale numeric features practically to [0,1] because nonnegativity
    #     columns=["game_age_inv"],
    #     index=df.index,
    # )

    # age_inv = sparse.csr_matrix(age_inv)

    # apply weights to each sparse feature group
    # this sets the importance of each feature group in the cosine similarity calculation
    # this has to be applied after normalization so that the weights actually affect the similarity scores
    blocks: list[sparse.csr_matrix] = [
        normalize(tags, norm="l2", axis=1) * config["w_tags"],
        # age_inv * config["w_age"],
    ]

    X = sparse.hstack(blocks, format="csr")
    X = normalize(X, norm="l2", axis=1)  # rescales vectors to unit length for cosine similarity
    items_df_modified = df.reset_index(drop=True)  # indexed by row number which corresponds to the item_matrix rows
    item_matrix = X  # shape (n_items, n_features)
    item_ids = df["item_id"].astype(int).to_numpy()
    item_id_to_idx = {int(item_id): idx for idx, item_id in enumerate(item_ids)}  # mapping from item_id to row index in item_matrix

    print(f"Item matrix shape: {item_matrix.shape[0]:,} x {item_matrix.shape[1]:,} (# of games x # of tags)")

    return items_df_modified, item_matrix, item_id_to_idx


# this build item score for re-ranking based on popularity, quality, and recency
def build_item_score(items_df: pd.DataFrame) -> pd.DataFrame:
    df = items_df[["item_id", "user_reviews", "rating", "game_age_inv"]].copy()

    df["user_reviews"] = pd.to_numeric(df["user_reviews"], errors="coerce").fillna(int(0))  # fill missing review counts with zero
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(int(0))  # fill missing ratings 0
    df["game_age_inv"] = pd.to_numeric(df["game_age_inv"], errors="coerce")  # inverse of age to give more weight to newer games

    # popularity
    df["pop_score"] = _safe_minmax(np.log1p(df["user_reviews"]))  # apply log transformation

    # quality
    df["quality_score"] = _safe_minmax(df["rating"])  # normalize to [0,1] range

    # normalized game age
    df["age_score"] = _safe_minmax(df["game_age_inv"])  # normalize to [0,1] range

    return df[["item_id", "pop_score", "quality_score", "age_score"]]


def build_als_matrix(
    interactions_df: pd.DataFrame,
    alpha: float = config["als_alpha"],
) -> tuple[sparse.csr_matrix, dict[int, int], dict[int, int]]:
    """
    Builds a (n_users x n_items) sparse CSR confidence matrix for ALS.
    confidence[u, i] = 1 + alpha * interaction_score(u, i)

    Returns:
        confidence_matrix : sparse.csr_matrix, shape (n_users, n_items)
        user_id_to_idx    : dict mapping steam_id -> row index
        item_id_to_idx_als: dict mapping item_id  -> col index
    """

    user_ids = interactions_df["steam_id"].unique()
    item_ids = interactions_df["item_id"].unique()

    user_id_to_idx_als = {int(uid): idx for idx, uid in enumerate(user_ids)}  # map steam_id to row index
    item_id_to_idx_als = {int(iid): idx for idx, iid in enumerate(item_ids)}  # map item_id to col index

    rows = interactions_df["steam_id"].map(user_id_to_idx_als).to_numpy()
    cols = interactions_df["item_id"].map(item_id_to_idx_als).to_numpy()

    # confidence = 1 + alpha * interaction_score
    # interaction_score already encodes log(hours) and recommend,
    # so just need to scale it by alpha and add 1 for the baseline confidence
    data = (1.0 + alpha * interactions_df["interaction_score"]).to_numpy(dtype=np.float32)

    n_users = len(user_ids)
    n_items = len(item_ids)

    confidence_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)  # build the sparse matrix in COO format and convert to CSR

    print(f"ALS matrix: {n_users:,} users x {n_items:,} items with {confidence_matrix.nnz:,} interactions")

    return confidence_matrix, user_id_to_idx_als, item_id_to_idx_als


# training the ALS model with the confidence matrix built from the interactions
def train_als(
    confidence_matrix: sparse.csr_matrix,
    factors: int = config["als_factors"],
    iterations: int = config["als_iterations"],
    regularization: float = config["als_regularization"],
    alpha: float = config["als_alpha"],
    random_state: int = config["random_state"],
) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
    )


    model.fit(confidence_matrix)  # expects items x users matrix
    print(
        "ALS trained: {factors} factors, {iterations} iterations, reg={reg}, alpha={alpha}".format(
            factors=factors,
            iterations=iterations,
            reg=regularization,
            alpha=alpha,
        )
    )

    return model


def main() -> None:
    args = parse_args()
    engine = create_engine(get_db_url())  # connect database

    print("")
    print("Loading items from database...")
    items_df = load_items(engine)
    items_df = filter_sparse_items(items_df)
    items_df, item_matrix, _ = fit_items(items_df)
    item_rerank_df = build_item_score(items_df)

    print("")
    print("Loading interactions from database...")
    interactions_df = load_interactions(engine)  # run for ~4.5 mins
    interactions_df = filter_interactions_to_runtime_items(interactions_df, items_df)

    print("")
    print("Training ALS model...")
    confidence_matrix, user_id_to_idx_als, item_id_to_idx_als = build_als_matrix(interactions_df)
    als_model = train_als(confidence_matrix)  # run for ~15 mins

    # # save only the interaction score per user-item pairs
    # interaction_scores_df = interactions_df[["steam_id", "item_id", "interaction_score"]].copy()
    # interaction_scores_df["steam_id"] = interaction_scores_df["steam_id"].astype("int64")
    # interaction_scores_df["item_id"] = interaction_scores_df["item_id"].astype("int64")
    # interaction_scores_df["interaction_score"] = interaction_scores_df["interaction_score"].astype("float32")

    print("")
    print("Saving CSV files for recommender service...")
    print("")

    # ======================
    # not saving user-related matrices -> steam_id from live API, compute on the fly
    # ======================

    # print("Saving interaction scores...")
    # save_csv(
    #     interaction_scores_df,
    #     args.output_dir / "interaction_scores.csv",
    # )
    print("Saving re-rank scores...")
    save_csv(
        item_rerank_df,
        args.output_dir / "item_rerank_scores.csv",
    )
    # print("Saving ALS user factors...")
    # save_factor_csv(
    #     args.output_dir / "als_user_factors.csv",
    #     user_id_to_idx_als,
    #     als_model.user_factors,
    #     "steam_id",
    #     "als_user_idx",
    # )
    print("Saving ALS item factors...")
    save_factor_csv(
        args.output_dir / "als_item_factors.csv",
        item_id_to_idx_als,
        als_model.item_factors,
        "item_id",
        "als_item_idx",
    )
    # print("Saving ALS user index...")
    # save_index_csv(
    #     args.output_dir / "als_user_index.csv",
    #     user_id_to_idx_als,
    #     id_column="steam_id",
    #     idx_column="als_user_idx",
    # )
    print("Saving ALS item index...")
    save_index_csv(
        args.output_dir / "als_item_index.csv",
        item_id_to_idx_als,
        id_column="item_id",
        idx_column="als_item_idx",
    )
    print("Saving item similarity scores...")
    save_item_similarity_csv( # this will save the top-k similar items for each item based on cosine similarity of the item feature matrix
        args.output_dir / "item_similarity.csv",
        items_df,
        item_matrix,
        args.item_similarity_top_k,  # top k similar items for each item
    )

    print(f"[csv] Wrote recommender CSVs to {args.output_dir}")

    print("")
    print("Saving runtime item artifacts for Flask recommender...")
    save_runtime_artifacts(
        args.runtime_output_dir,
        items_df=items_df,
        # item_rerank_df=item_rerank_df,
        item_factors=als_model.item_factors,
        item_id_to_idx_als=item_id_to_idx_als,
    )
    print(f"[runtime] Wrote runtime artifacts to {args.runtime_output_dir}")



    # run
    # robocopy "C:\Users\salir\OneDrive\Documents\Personal Project\Steam-Recommendation-System\tables\rec_matrices" "C:\ProgramData\MySQL\MySQL Server 8.0\Uploads" /E /Z /XA:H /W:5 /R:5

if __name__ == "__main__":
    main()
