#!/usr/bin/env python3

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1" # limit to single thread for ALS training

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.neighbors import NearestNeighbors
from implicit.cpu.als import AlternatingLeastSquares

from pathlib import Path
import gc
import csv
import json

try:
    from src.config import RECOMMENDER_CONFIG
    from src.helpers import _safe_minmax, save_csv, split_pipe_values
except ImportError:
    from config import RECOMMENDER_CONFIG
    from helpers import _safe_minmax, save_csv, split_pipe_values

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_GAME_DATA_FILE = BASE_DIR / "tables" / "production" / "GAME_DATA.csv"

OUTPUT_SIMILARITY_FILE = BASE_DIR / "tables" / "production" / "GAME_SIMILARITY.csv"
OUTPUT_RERANK_FILE = BASE_DIR / "tables" / "production" / "GAME_SCORES.csv"
OUTPUT_GAME_FACTORS_DIR = BASE_DIR / "tables" / "production"



def build_feature_frame(base_dir: Path = BASE_DIR) -> tuple[pd.DataFrame, object]:
    game_data = pd.read_csv(INPUT_GAME_DATA_FILE)
    df = game_data.copy()

    for cols in ["developers", "publishers", "tags"]:
        df[cols] = df[cols].apply(split_pipe_values) # replace missing lists with empty

    # combine all sparse categorical features into one list
    df["all_features"] = (
        df["developers"].apply(lambda values: [f"dev:{x}" for x in values]) # prefix dev: to distinguish from publishers and tags
        + df["publishers"].apply(lambda values: [f"pub:{x}" for x in values]) # prefix pub: to distinguish from developers and tags
        + df["tags"].apply(lambda values: [f"tag:{x}" for x in values]) # prefix tag: to distinguish from developers and publishers
    )

    # each column is a feature (dev, pub, or tag) and each row is an item; value is 1 if item has that feature, else 0
    # binary encoding of multi-label features into a sparse matrix
    mlb = MultiLabelBinarizer(sparse_output=True)
    feature_matrix = mlb.fit_transform(df["all_features"]) # sparse binary matrix where rows are items and columns are features (devs, pubs, tags)
    X = normalize(feature_matrix, norm="l2", axis=1) # normalize rows to unit length for cosine similarity

    return df, X


def build_score_frame(df: pd.DataFrame) -> pd.DataFrame:
    score_df = df[["item_id", "user_reviews", "rating", "release_date"]].copy()

    dates = pd.to_datetime(score_df["release_date"], errors="coerce")
    current_year = pd.Timestamp.now().year
    release_year = dates.dt.year.fillna(2003).astype(int) # 2003 Steam founded
    score_df["age_inv"] = 1 / (current_year - release_year + 1) # inverse of age to give more weight to newer games

    # all is normalized to [0,1] range
    score_df["pop_score"] = _safe_minmax(np.log1p(score_df["user_reviews"])) # for popularity score; log transform to reduce skew
    score_df["quality_score"] = _safe_minmax(score_df["rating"]) # for quality score
    score_df["age_score"] = _safe_minmax(score_df["age_inv"]) # for age score; newer games get higher score

    score_df = score_df[["item_id", "pop_score", "quality_score", "age_score"]]
    return score_df

def save_item_similarity_csv(
    output_path: Path,
    items_df: pd.DataFrame,
    item_matrix,
    candidate_pool_size: int = RECOMMENDER_CONFIG["candidate_pool_size"], # how many similar items to save per item (excluding itself)
    batch_size: int = 10_000,
) -> None:
    """
    Compute top-k item neighbors in batches and save them to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    item_ids = items_df["item_id"].astype(int).tolist()
    n_items = len(item_ids)

    # check for empty catalogue; should not happen
    # if n_items == 0 or top_k <= 0:
    #     with output_path.open("w", encoding="utf-8", newline="") as handle:
    #         writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    #         writer.writerow(["source_item_id", "similar_item_id", "similarity_rank", "similarity_score"])
    #     return

    # use one extra neighbor because the query item itself is returned
    n_neighbors = min(candidate_pool_size + 1, n_items)

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(item_matrix) # fit once on the full item matrix

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        writer.writerow(["source_item_id", "similar_item_id", "similarity_score"]) # the columns

        # batching to reduce memory usage and I/O overhead; process items in chunks
        for start_idx in range(0, n_items, batch_size):
            end_idx = min(start_idx + batch_size, n_items)
            distances, indices = nn.kneighbors(item_matrix[start_idx:end_idx], return_distance=True) # get neighbors for the batch of items

            batch_rows: list[list[object]] = []
            for batch_offset, source_item_id in enumerate(item_ids[start_idx:end_idx]):
                neighbor_pairs: list[tuple[int, float]] = []

                for distance, similar_idx in zip(distances[batch_offset], indices[batch_offset]):
                    similar_idx = int(similar_idx)
                    if similar_idx == start_idx + batch_offset:
                        continue # skip self-match

                    # distances are cosine distances; convert to similarity
                    similarity = 1.0 - float(distance)
                    neighbor_pairs.append((similar_idx, similarity))
                    if len(neighbor_pairs) == candidate_pool_size: # only keep top_k neighbors
                        break

                for similar_idx, similarity in neighbor_pairs:
                    batch_rows.append(
                        [
                            int(source_item_id), # source item id
                            int(item_ids[similar_idx]), # similar item id
                            similarity, # similarity score (cosine similarity in [0,1])
                        ]
                    )

            writer.writerows(batch_rows)
            print(f"Saved similarity rows for items {start_idx + 1:,}-{end_idx:,} of {n_items:,}")

def build_review_matrix(
    catalogue_df: pd.DataFrame,
    base_dir: Path = BASE_DIR,
) -> tuple[sparse.csr_matrix, dict[int, int]]:
    review_data = pd.read_csv(base_dir / "tables" / "production" / "GAME_REVIEW.csv")

    catalogue_median = catalogue_df[["item_id", "median_playtime_forever"]].copy()
    catalogue_median["item_id"] = pd.to_numeric(catalogue_median["item_id"], errors="coerce")
    catalogue_median["median_playtime_forever"] = pd.to_numeric(catalogue_median["median_playtime_forever"],errors="coerce")
    review_data = review_data.merge(catalogue_median, on="item_id", how="left")

    review_data["hours"] = pd.to_numeric(review_data["hours"], errors="coerce").fillna(0.0)
    review_data["recommendation"] = pd.to_numeric(review_data["recommendation"], errors="coerce").fillna(1)
    review_data["early_access"] = pd.to_numeric(review_data["early_access"], errors="coerce").fillna(0)
  

    # IMPORTANT!
    # normalize playtime hours by the game's median lifetime playtime so shorter games are not penalized
    median_hours = np.maximum(review_data["median_playtime_forever"].fillna(1.0).to_numpy(dtype=np.float32), 1.0)
    normalized_hours = review_data["hours"].to_numpy(dtype=np.float32) / median_hours

    review_data["interaction_score"] = (  # calculate interaction score based on recommend and hours
        np.log1p(normalized_hours)  # scale hours relative to the game's typical playtime before logging
        * review_data["recommendation"].map({2: 1.5, 1: 1.0, 0: 0.5})  # give more weight to recommend=2 and less weight to recommend=0
        * review_data["early_access"].map({1: 0.8, 0: 1.0}) # give less weight to early access reviews
    ).astype(np.float32)

    # filter to only items in the catalogue
    valid_item_ids = set(pd.to_numeric(catalogue_df["item_id"], errors="coerce").dropna().astype(int).tolist()) # get unique valid item ids
    review_data = review_data[review_data["item_id"].isin(valid_item_ids)]

    user_ids = review_data["steam_id"].unique()
    item_ids = review_data["item_id"].unique()

    user_id_to_idx_als = {int(uid): idx for idx, uid in enumerate(user_ids)}
    item_id_to_idx_als = {int(iid): idx for idx, iid in enumerate(item_ids)}

    rows = review_data["steam_id"].map(user_id_to_idx_als).to_numpy()  # map steam_id to row index
    cols = review_data["item_id"].map(item_id_to_idx_als).to_numpy()  # map item_id to col index

    # confidence = 1 + alpha * interaction_score
    # see http://yifanhu.net/PUB/cf.pdf
    data = (1.0 + RECOMMENDER_CONFIG["als_alpha"] * review_data["interaction_score"]).to_numpy(dtype=np.float32)

    n_users = len(user_ids)
    n_items = len(item_ids)

    confidence_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)  # build the sparse matrix in COO format and convert to CSR
    return confidence_matrix, item_id_to_idx_als

# train the ALS model on the confidence matrix and return the trained model
# factors and hyperparameters are set in config.py
def train_als(confidence_matrix: sparse.csr_matrix) -> AlternatingLeastSquares:
    factors: int = RECOMMENDER_CONFIG["als_factors"]
    iterations: int = RECOMMENDER_CONFIG["als_iterations"]
    regularization: float = RECOMMENDER_CONFIG["als_regularization"]
    alpha: float = RECOMMENDER_CONFIG["als_alpha"]
    random_state: int = RECOMMENDER_CONFIG["random_state"]
    model = AlternatingLeastSquares( # ALS hyperparameters; set in config.py
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
    )
    print("Training ALS model on confidence matrix with shape {}...".format(confidence_matrix.shape))
    model.fit(confidence_matrix)  # expects users x items matrix
    print(
        "ALS trained: {factors} factors, {iterations} iterations, reg={reg}, alpha={alpha}".format(
            factors=factors,
            iterations=iterations,
            reg=regularization,
            alpha=alpha,
        )
    )
    return model


def save_runtime_artifacts(
    output_dir: Path,
    item_factors: np.ndarray,
    item_id_to_idx_als: dict[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "item_factors.npy", np.asarray(item_factors, dtype=np.float32))
    with (output_dir / "item_id_to_idx_als.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {str(int(item_id)): int(idx) for item_id, idx in sorted(item_id_to_idx_als.items(), key=lambda item: item[1])},
            handle,
            ensure_ascii=True,
            indent=2,
        )



def main() -> None:
    df, X = build_feature_frame() # item metadata and the sparse matrix


    score_df = build_score_frame(df)
    save_item_similarity_csv(OUTPUT_SIMILARITY_FILE, df, X)
    save_csv(score_df, OUTPUT_RERANK_FILE)

    confidence_matrix, item_id_to_idx_als = build_review_matrix(df) # from GAME_REVIEW.csv

    del df
    gc.collect()

    model = train_als(confidence_matrix)
    # this won't be saved to SQL because we need the full vectors to compute similarities; see recommender.py
    save_runtime_artifacts(
        OUTPUT_GAME_FACTORS_DIR,
        item_factors=model.item_factors,
        item_id_to_idx_als=item_id_to_idx_als,
    )
    print(f"Shape of item factors: {model.item_factors.shape}")
    print(f"[runtime] Wrote runtime artifacts to {OUTPUT_GAME_FACTORS_DIR}")

if __name__ == "__main__":
    main()
