"""
Recommender service implementation.
This module provides two recommenders:
1. Collaborative filtering (CF) via ALS fold-in from live user interactions.
2. Content-based (CB) recommendations via precomputed item similarity.

Both recommenders use the same item-level re-rank scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text, create_engine

try:
    from src.config import get_recommender_config, get_db_url, QUERY_CONFIG
    from src.helpers import _safe_minmax, to_list
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution inside src/
    from config import get_recommender_config, get_db_url, QUERY_CONFIG
    from helpers import _safe_minmax, to_list


@dataclass
class RuntimeArtifacts:
    items_df: pd.DataFrame
    item_factors: np.ndarray
    item_id_to_idx_als: dict[int, int]
    item_factor_gram: np.ndarray


config = get_recommender_config()
engine = create_engine(get_db_url())
_RUNTIME_CACHE: dict[Path, RuntimeArtifacts] = {}


def _load_json_mapping(path: Path) -> dict[int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): int(value) for key, value in raw.items()}


def _merge_weights(overrides: dict[str, float] | None = None) -> dict[str, float]:
    merged = config.copy()
    if overrides:
        merged.update(overrides) # if weights overrides are provided, use these
    return merged # if not, use default from config


# fetch item factors for a list of item_ids from the database
# these factors are used for folding in the user vector in ALS
def fetch_als_item_factors(engine, item_id_list: list[int]) -> pd.DataFrame:
    if not item_id_list:
        return pd.DataFrame(columns=["item_id", "als_item_idx", "factors"])

    query = text(f"""
        SELECT *
        FROM {QUERY_CONFIG["database_query_name"]}.als_item_factors
        """) # .bindparams(bindparam("item_id_list", expanding=True))
        # -- WHERE item_id IN :item_id_list

    result = pd.read_sql(query, engine, )
                         # params={"item_id_list": [int(x) for x in item_id_list]})
    if result.empty:
        return pd.DataFrame(columns=["item_id", "als_item_idx", "factors"])

    result["item_id"] = result["item_id"].astype(np.int32)
    result["als_item_idx"] = result["als_item_idx"].astype(np.int32)
    result["factors"] = result["factors"].apply(lambda x: np.asarray(json.loads(x), dtype=np.float32)) # convert JSON string back to numpy array
    return result


def fetch_item_rerank_scores(engine, item_id_list: list[int]) -> pd.DataFrame:
    if not item_id_list:
        return pd.DataFrame(columns=["item_id", "pop_score", "quality_score", "age_score"])

    query = text(f"""
        SELECT *
        FROM {QUERY_CONFIG["database_query_name"]}.item_rerank_scores
        WHERE item_id IN :item_id_list
    """).bindparams(bindparam("item_id_list", expanding=True))

    result = pd.read_sql(query, engine, params={"item_id_list": [int(x) for x in item_id_list]})
    if result.empty:
        return pd.DataFrame(columns=["item_id", "pop_score", "quality_score", "age_score"])

    result["item_id"] = result["item_id"].astype(np.int32)
    result["pop_score"] = pd.to_numeric(result["pop_score"], errors="coerce").fillna(0.0).astype(np.float32)
    result["quality_score"] = pd.to_numeric(result["quality_score"], errors="coerce").fillna(0.0).astype(np.float32)
    result["age_score"] = pd.to_numeric(result["age_score"], errors="coerce").fillna(0.0).astype(np.float32)
    return result


# load runtime from MySQL query (for CF only)
def load_runtime_mysql(output_dir: Path) -> RuntimeArtifacts:
    """
    Fetch all necessary runtime artifacts for the recommenders, including item data, ALS item factors, item re-rank scores, and precomputed item factor gram matrix.
    The artifacts will be cached in memory after the first load for efficiency.
    """

    items_df = pd.read_sql(text(QUERY_CONFIG["item_query"]), engine)
    items_df["tags"] = items_df["tags"].apply(to_list)


    item_factors_df = fetch_als_item_factors(engine=engine, item_id_list=items_df["item_id"].tolist())
    item_factors_df = item_factors_df.sort_values("als_item_idx").reset_index(drop=True) # ensure the item factors are ordered by ALS index for correct matrix operations
    item_factors = np.vstack(item_factors_df["factors"].to_numpy()).astype(np.float32) # convert list of factor arrays into a 2D numpy array
    item_id_to_idx_als = item_factors_df.set_index("item_id")["als_item_idx"].astype(int).to_dict() # create mapping from item_id to ALS item index for quick lookup during fold-in
    item_factor_gram = item_factors.T @ item_factors # precompute the gram matrix of item factors for efficient ALS fold-in computations


    runtime = RuntimeArtifacts(
        items_df=items_df,
        item_factors=item_factors,
        item_id_to_idx_als=item_id_to_idx_als,
        item_factor_gram=item_factor_gram,
    )
    _RUNTIME_CACHE[output_dir] = runtime
    return runtime


# load runtime from MySQL query from saved files (for CF only)
def load_runtime(output_dir: Path) -> RuntimeArtifacts:
    output_dir = output_dir.resolve()
    cached = _RUNTIME_CACHE.get(output_dir)
    if cached is not None:
        return cached

    items_df = pd.read_parquet(output_dir / "items_used.parquet").copy()
    item_factors = np.load(output_dir / "item_factors.npy").astype(np.float32)
    item_id_to_idx_als = _load_json_mapping(output_dir / "item_id_to_idx_als.json")
    item_factor_gram = np.asarray(item_factors.T @ item_factors, dtype=np.float32)

    runtime = RuntimeArtifacts(
        items_df=items_df,
        item_factors=item_factors,
        item_id_to_idx_als=item_id_to_idx_als,
        item_factor_gram=item_factor_gram,
    )
    _RUNTIME_CACHE[output_dir] = runtime
    return runtime


# =========================================
# Collaborative-filtering Recommender
# =========================================



# compute final score of each filtered games
# the recommendation list will then be re-ranked based on the final score
def _apply_shared_rerank(
    candidates_df: pd.DataFrame,
    base_score_column: str,
    base_weight_key: str,
    weight_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    weights = _merge_weights(weight_overrides)
    candidates = candidates_df.copy()

    candidates[base_score_column] = _safe_minmax(candidates[base_score_column])
    candidates["pop_score"] = candidates["pop_score"].fillna(0.0)
    candidates["quality_score"] = candidates["quality_score"].fillna(0.0)
    candidates["age_score"] = candidates["age_score"].fillna(0.0)

    candidates["final_score"] = (
        weights[base_weight_key] * candidates[base_score_column] # based on CF score or similarity score
        + weights["w_popularity"] * candidates["pop_score"] # based on user reviews
        + weights["w_quality"] * candidates["quality_score"] # based on positive reviews
        + weights["w_age"] * candidates["age_score"] # based on release date, newer is better
    )
    return candidates

# rerank candidates based on the final score after weighting
def rerank_candidates(
    candidates_df: pd.DataFrame,
    base_score_column: str,
    base_weight_key: str,
    top_n: int = 15,
    weight_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    candidates = _apply_shared_rerank(
        candidates_df=candidates_df,
        base_score_column=base_score_column,
        base_weight_key=base_weight_key,
        weight_overrides=weight_overrides,
    )
    candidates = candidates.sort_values(
        ["final_score", base_score_column, "item_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return candidates.head(top_n)


# =========================================
# Collaborative-filtering Recommender
# =========================================


def build_interaction_scores(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build interaction scores from live user data containing item ownership interactions.

    Expected columns:
        item_id
        hours
        recommend

    Output columns:
        item_id
        interaction_score
    """

    if interactions_df.empty:
        return pd.DataFrame(columns=["item_id", "interaction_score"])

    df = interactions_df.copy()
    # if "interaction_score" in df.columns and "hours" not in df.columns:
    #     df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce")
    #     df["interaction_score"] = pd.to_numeric(df["interaction_score"], errors="coerce").fillna(0.0)
    #     df = df.dropna(subset=["item_id"])
    #     df = df[df["interaction_score"] > 0].copy()
    #     if df.empty:
    #         return pd.DataFrame(columns=["item_id", "interaction_score"])

    #     df["item_id"] = df["item_id"].astype(np.int64)
    #     df = (
    #         df.sort_values(["item_id", "interaction_score"], ascending=[True, False])
    #         .drop_duplicates(subset=["item_id"], keep="first")
    #         .reset_index(drop=True)
    #     )
    #     return df[["item_id", "interaction_score"]]

    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce")
    df["hours"] = pd.to_numeric(df.get("hours"), errors="coerce").fillna(0.0)
    df["recommend"] = pd.to_numeric(df.get("recommend"), errors="coerce").fillna(0)
    df = df.dropna(subset=["item_id"])

    # compute interaction score following log(1 + hours) * (1.5 if recommend=1 else 0.5 if recommend=0 else 0)
    df["interaction_score"] = (
        np.log1p(df["hours"])
        * df["recommend"].map({1: 1.5, 0: 0.5}).fillna(0.0)
    ).astype(np.float32)
    df = df[df["interaction_score"] > 0].copy() # exclude meaningless interactions

    if df.empty:
        return pd.DataFrame(columns=["item_id", "interaction_score"])

    df["item_id"] = df["item_id"].astype(np.int64)
    # df = ( #  drop duplicates of item_id, keep the one with highest interaction_score; there should be no duplicates from live API
    #     df.sort_values(["item_id", "interaction_score"], ascending=[True, False])
    #     .drop_duplicates(subset=["item_id"], keep="first")
    #     .reset_index(drop=True)
    # )
    return df[["item_id", "interaction_score"]]



def _fold_in_user_vector(
    runtime: RuntimeArtifacts,
    scored_interactions_df: pd.DataFrame,
    alpha: float = config["als_alpha"],
    regularization: float = config["als_regularization"],
) -> tuple[np.ndarray | None, set[int]]:
    
    observed = scored_interactions_df[["item_id", "interaction_score"]].copy() # compute the interaction score
    if observed.empty:
        return None, set()
    observed["item_id"] = observed["item_id"].astype(int)
    observed["interaction_score"] = pd.to_numeric(observed["interaction_score"],errors="coerce",).fillna(0.0)

    # get the mapping from item_id to als_item_idx
    # user_item_factors = fetch_als_item_factors(engine=engine, item_id_list=observed["item_id"].tolist())
    observed["als_item_idx"] = observed["item_id"].map(runtime.item_id_to_idx_als)
    # observed = observed.merge(item_factors_mapping, on="item_id", how="left")
    observed = observed.dropna(subset=["als_item_idx"]).copy()
    if observed.empty:
        return None, set()

    seen_item_ids = set(observed["item_id"].astype(int).tolist()) # keep track of items the user has interacted with, to exclude from recommendations
    observed_idxs = observed["als_item_idx"].astype(int).to_numpy()
    observed_vectors = runtime.item_factors[observed_idxs]
    # observed_vectors = np.vstack(user_item_factors["factors"].to_numpy())

    # confidence computation; see http://yifanhu.net/PUB/cf.pdf equation (6)
    confidences = (1.0 + alpha * observed["interaction_score"].to_numpy(dtype=np.float32)).astype(np.float32)

    # compute user vector via ALS fold-in; see http://yifanhu.net/PUB/cf.pdf equation (4)
    a_matrix = np.array(runtime.item_factor_gram, dtype=np.float32, copy=True)
    a_matrix += regularization * np.eye(a_matrix.shape[0], dtype=np.float32)
    b_vector = np.zeros(a_matrix.shape[0], dtype=np.float32)

    for confidence, item_vector in zip(confidences, observed_vectors):
        scaled_confidence = float(confidence - 1.0)
        if scaled_confidence > 0:
            a_matrix += scaled_confidence * np.outer(item_vector, item_vector)
        b_vector += float(confidence) * item_vector

    return np.linalg.solve(a_matrix, b_vector).astype(np.float32), seen_item_ids



def build_cf_candidates(
    live_interactions_df: pd.DataFrame,
    runtime: RuntimeArtifacts,
    candidate_pool_size: int = 200,
) -> pd.DataFrame:
    """
    Build the CF candidate list up to the point just before applying adjustable weights.
    The returned DataFrame already includes the shared re-rank feature columns.
    """

    scored_interactions_df = build_interaction_scores(live_interactions_df) # columns item_id and interaction_score
    if scored_interactions_df.empty:
        return pd.DataFrame(columns=["item_id", "cf_score", "pop_score", "quality_score", "age_score"])

    user_vector, seen_item_ids = _fold_in_user_vector(runtime, scored_interactions_df) # fold-in user vector based on interaction scores and item factors
    if user_vector is None:
        return pd.DataFrame(columns=["item_id", "cf_score", "pop_score", "quality_score", "age_score"])

    scores = runtime.item_factors @ user_vector # compute CF scores for all items by taking dot product between item factors and user vector
    idx_to_item_id = { # create reverse mapping from ALS item index to item_id
        factor_idx: item_id for item_id, factor_idx in runtime.item_id_to_idx_als.items()
    }
    ordered_item_ids = [idx_to_item_id[idx] for idx in range(len(idx_to_item_id))]
    candidates = pd.DataFrame({"item_id": ordered_item_ids}) # create dataframe of candidate items with their item_ids ordered by ALS index;
    runtime_item_ids = set(pd.to_numeric(runtime.items_df["item_id"], errors="coerce").dropna().astype(int).tolist())
    candidates = candidates[candidates["item_id"].isin(runtime_item_ids)].copy()
    # candidates = candidates.merge(
    #     runtime.items_df[["item_id", "item_name"]],
    #     on="item_id",
    #     how="left",
    # )
    candidates["cf_score"] = scores.astype(np.float32)
    if seen_item_ids:
        candidates = candidates[~candidates["item_id"].isin(seen_item_ids)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["item_id", "cf_score", "pop_score", "quality_score", "age_score"])

    pool_size = max(int(candidate_pool_size), 1)
    candidates = candidates.nlargest(pool_size, "cf_score").copy() # keep top-N candidates based on CF score before merging with re-rank scores

    rerank_df = fetch_item_rerank_scores(
        engine=engine,
        item_id_list=candidates["item_id"].astype(int).tolist(),
    )
    candidates = candidates.merge(rerank_df, on="item_id", how="left") # merge with item-level re-rank scores
    return candidates[["item_id", "cf_score", "pop_score", "quality_score", "age_score"]]

# apply the weights after re-rank merging, and compute the final score for each candidate
def recommend_cf(
    live_interactions_df: pd.DataFrame, # dataframe whose data got fetched from live API
    runtime: RuntimeArtifacts,
    top_n: int = 15,
    candidate_pool_size: int = 200,
    weight_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Collaborative-filtering recommender using ALS fold-in from online user interactions.
    """
    candidates = build_cf_candidates(
        live_interactions_df=live_interactions_df,
        runtime=runtime,
        candidate_pool_size=max(candidate_pool_size, top_n),
    )
    if candidates.empty:
        return pd.DataFrame(columns=["item_id", "final_score"])

    ranked = rerank_candidates(
        candidates_df=candidates,
        base_score_column="cf_score",
        base_weight_key="w_cf",
        top_n=top_n,
        weight_overrides=weight_overrides,
    )
    return ranked[["item_id", "final_score"]]


# # functions below are not used since user fetching is done through live API

# def fetch_user_interactions(
#     engine,
#     steam_id: int,
#     table_name: str | None = None,
# ) -> pd.DataFrame:
#     qualified_table_name = table_name or _table_name("user_interactions_table")
#     query = text(
#         f"""
#         SELECT
#             r.steam_id,
#             r.item_id,
#             r.interaction_score
#         FROM {qualified_table_name} r
#         WHERE r.steam_id = :steam_id
#         """
#     )

#     interactions_df = pd.read_sql(query, engine, params={"steam_id": int(steam_id)})
#     if interactions_df.empty:
#         return pd.DataFrame(columns=["steam_id", "item_id", "interaction_score"])

#     interactions_df["steam_id"] = interactions_df["steam_id"].astype(np.int64)
#     interactions_df["item_id"] = interactions_df["item_id"].astype(np.int64)
#     interactions_df["interaction_score"] = pd.to_numeric(
#         interactions_df["interaction_score"],
#         errors="coerce",
#     ).fillna(0.0).astype(np.float32)
#     interactions_df = interactions_df[interactions_df["interaction_score"] > 0].copy()
#     return interactions_df.reset_index(drop=True)


# def fetch_user_interactions(
#     engine,
#     steam_id: int,
#     table_name: str | None = None,
# ) -> pd.DataFrame:

#     query = text(
#         f"""
#         SELECT
#             r.steam_id,
#             r.item_id,
#             r.recommend,
#             r.hours
#         FROM `steam-rec`.GAME_REVIEW r
#         WHERE r.steam_id = :steam_id
#         """
#     )

#     interactions_df = pd.read_sql(query, engine, params={"steam_id": int(steam_id)})
#     if interactions_df.empty:
#         return pd.DataFrame(columns=["steam_id", "item_id", "recommend", "hours"])

#     interactions_df["steam_id"] = pd.to_numeric(
#         interactions_df["steam_id"],
#         errors="coerce",
#     ).astype("Int64")
#     interactions_df["item_id"] = pd.to_numeric(
#         interactions_df["item_id"],
#         errors="coerce",
#     ).astype("Int64")
#     interactions_df["recommend"] = pd.to_numeric(
#         interactions_df["recommend"],
#         errors="coerce",
#     ).fillna(0).astype(np.int8)
#     interactions_df["hours"] = pd.to_numeric(
#         interactions_df["hours"],
#         errors="coerce",
#     ).fillna(0.0).astype(np.float32)
#     return interactions_df.reset_index(drop=True)


# def recommend_from_liked_games(
#     interactions_df: pd.DataFrame | None = None, # MUST consists of item_id, hours, and recommend columns
#     runtime: RuntimeArtifacts | None = None,
#     live_interactions_df: pd.DataFrame | None = None,
#     top_n: int = 15,
#     weight_overrides: dict[str, float] | None = None,
#     **_: Any,
# ) -> pd.DataFrame:
#     if runtime is None:
#         raise ValueError("runtime is required")

#     source_df = live_interactions_df if live_interactions_df is not None else interactions_df
#     if source_df is None:
#         return pd.DataFrame(columns=["item_id", "item_name", "final_score"])

#     result_size = top_n

#     # if "interaction_score" in source_df.columns and "hours" not in source_df.columns:
#     #     cf_input_df = source_df[["item_id", "interaction_score"]].copy()
#     # else:
#     #     cf_input_df = source_df.copy()

#     return recommend_cf(
#         live_interactions_df=source_df,
#         runtime=runtime,
#         top_n=result_size,
#         weight_overrides=weight_overrides,
#     )




# =========================================
# Content-based Recommender
# =========================================

def fetch_item_similarity(
    engine,
    item_id: int,
    top_n: int = 140,
) -> pd.DataFrame:

    # joining item similarity matrix, item re-rank matrix, and game data table
    query = text(
        f"""
        SELECT
            r.source_item_id,
            r.similar_item_id AS item_id,
            g.item_name,
            -- r.similarity_rank,
            r.similarity_score,
    	    rk.pop_score,
	        rk.quality_score,
	        rk.age_score
        FROM {QUERY_CONFIG["database_query_name"]}.item_similarity r
        JOIN {QUERY_CONFIG["database_prod_name"]}.GAME_DATA g ON g.item_id = r.similar_item_id
        JOIN {QUERY_CONFIG["database_query_name"]}.item_rerank_scores rk ON rk.item_id = r.similar_item_id
        WHERE r.source_item_id = :item_id
        ORDER BY r.similarity_rank ASC -- rank #1 is most similar
        LIMIT :top_n
        """
    )
    result = pd.read_sql(
        query,
        engine,
        params={"item_id": int(item_id), "top_n": int(top_n)},
    )
    if result.empty:
        return pd.DataFrame(columns=["source_item_id", "item_id", "item_name",
                                     "similarity_score", "pop_score", "quality_score", "age_score"])

    result["source_item_id"] = result["source_item_id"].astype(np.int64)
    result["item_id"] = result["item_id"].astype(np.int64)
    # result["similarity_rank"] = result["similarity_rank"].astype(np.int64)

    result["similarity_score"] = pd.to_numeric(result["similarity_score"],errors="coerce",).fillna(0.0).astype(np.float32)
    result["pop_score"] = pd.to_numeric(result["pop_score"],errors="coerce",).fillna(0.0).astype(np.float32)
    result["quality_score"] = pd.to_numeric(result["quality_score"],errors="coerce",).fillna(0.0).astype(np.float32)
    result["age_score"] = pd.to_numeric(result["age_score"],errors="coerce",).fillna(0.0).astype(np.float32)

    return result


def build_cb_candidates(
    item_id: int,
    engine,
    similarity_pool_size: int = 140, # how many similar items to fetch before re-ranking (the more the better chance of good recommendations)
) -> pd.DataFrame:
    """
    Build the CB candidate list up to the point just before applying adjustable weights.
    The returned DataFrame already includes the shared re-rank feature columns.
    """
    candidates = fetch_item_similarity(engine=engine, item_id=item_id, top_n=similarity_pool_size)
    if candidates.empty:
        return pd.DataFrame(columns=["item_id", "similarity_score", "pop_score", "quality_score", "age_score"])
    return candidates[["item_id", "similarity_score", "pop_score", "quality_score", "age_score"]]


def recommend_cb(
    item_id: int,
    engine,
    top_n: int = 15,
    similarity_pool_size: int = 100,
    weight_overrides: dict[str, float] | None = None,
    **_: Any,
) -> pd.DataFrame:
    """
    Content-based recommender using precomputed item-to-item similarity.
    """
    candidates = build_cb_candidates(
        item_id=item_id,
        engine=engine,
        similarity_pool_size=similarity_pool_size,
    )
    if candidates.empty:
        return pd.DataFrame(columns=["item_id", "final_score"])

    # re-rank candidates
    ranked = rerank_candidates(
        candidates_df=candidates,
        base_score_column="similarity_score",
        base_weight_key="w_anchor_sim",
        top_n=top_n,
        weight_overrides=weight_overrides,
    )
    return ranked[["item_id", "final_score"]]


# =========================================
# WRAPPER FUNCTIONS FOR TESTS
# =========================================


def build_cf_candidates_from_liked_games(
    interactions_df: pd.DataFrame | None = None,
    runtime: RuntimeArtifacts | None = None,
    live_interactions_df: pd.DataFrame | None = None,
    candidate_pool_size: int = 200,
    **_: Any,
) -> pd.DataFrame:
    if runtime is None:
        raise ValueError("runtime is required")

    source_df = live_interactions_df if live_interactions_df is not None else interactions_df
    if source_df is None:
        return pd.DataFrame(columns=["item_id", "item_name", "cf_score", "pop_score", "quality_score", "age_score"])

    return build_cf_candidates(
        live_interactions_df=source_df,
        runtime=runtime,
        candidate_pool_size=candidate_pool_size,
    )



def build_cb_candidates_from_item(
    item_id: int,
    engine,
    similarity_pool_size: int = 100,
    **_: Any,
) -> pd.DataFrame:
    return build_cb_candidates(
        item_id=item_id,
        engine=engine,
        similarity_pool_size=similarity_pool_size,
    )
