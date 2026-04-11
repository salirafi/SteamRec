
#!usr/bin/env python3

import pandas as pd
from sqlalchemy import bindparam, text
import numpy as np

try:
    from src.config import DB_CONFIG, RECOMMENDER_CONFIG
    from src.helpers import _safe_minmax
except ImportError:
    from config import DB_CONFIG, RECOMMENDER_CONFIG
    from helpers import _safe_minmax



pool_size = RECOMMENDER_CONFIG["candidate_pool_size"]
top_n = RECOMMENDER_CONFIG["top_n"]
als_alpha = RECOMMENDER_CONFIG["als_alpha"]
als_regularization = RECOMMENDER_CONFIG["als_regularization"]

# ==================================
# HELPERS
# ==================================


def _apply_shared_rerank(
    candidates_df: pd.DataFrame,
    base_weight_key: str, # "w_cb" for content-based, "w_cf" for user-based
    # the key has to be [base_weight_key, "w_age", "w_popularity", "w_quality"]; see config.py
    weight_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    
    weights = {key: RECOMMENDER_CONFIG[key] for key in [base_weight_key, "w_age", "w_popularity", "w_quality"]}
    if weight_overrides:
        weights.update(weight_overrides)

    candidates = candidates_df.copy()

    candidates["similarity_score"] = _safe_minmax(candidates["similarity_score"])
    # the rest should already be normalized in recommender_matrices.py
    candidates["pop_score"] = candidates["pop_score"].fillna(0.0)
    candidates["quality_score"] = candidates["quality_score"].fillna(0.0)
    candidates["age_score"] = candidates["age_score"].fillna(0.0)

    candidates["final_score"] = (
        weights[base_weight_key] * candidates["similarity_score"] # based on CF or CB score
        + weights["w_popularity"] * candidates["pop_score"] # based on user reviews
        + weights["w_quality"] * candidates["quality_score"] # based on positive reviews
        + weights["w_age"] * candidates["age_score"] # based on release date, newer is better
    )
    return candidates

# rerank candidates based on the final score after weighting
def rerank_candidates(
    candidates_df: pd.DataFrame,
    base_weight_key: str,
    top_n: int = top_n,
    weight_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    candidates = _apply_shared_rerank(
        candidates_df=candidates_df,
        base_weight_key=base_weight_key,
        weight_overrides=weight_overrides,
    )
    candidates = candidates.sort_values(
        ["final_score", "item_id"],
        ascending=[False, True] # sort by final score descending, if tie then sort by item_id ascending for deterministic order
    ).reset_index(drop=True)
    return candidates.head(top_n)



# ==================================
# CONTENT-BASED RECOMMENDER
# ==================================


def fetch_item_similarity_cb(
    engine,
    item_id: int,
    candidate_pool_size: int = pool_size # this has to be smaller than "candidate_pool_size" in save_item_similarity_csv() in recommender_matrices.py
) -> pd.DataFrame:

    # MySQL query
    query = text(
        f"""
        SELECT
            r.source_item_id,
            r.similar_item_id AS item_id,
            g.item_name,
            g.release_date,
            g.user_reviews,
            g.rating,

            -- handle NULL as string
            COALESCE(g.tags, 'Unknown') AS tags,
            COALESCE(g.developers, 'Unknown') AS developers,
            COALESCE(g.publishers, 'Unknown') AS publishers,

            r.similarity_score,
            rk.pop_score,
            rk.quality_score,
            rk.age_score
        FROM {DB_CONFIG["database"]}.GAME_SIMILARITY r
        JOIN {DB_CONFIG["database"]}.GAME_DATA g ON g.item_id = r.similar_item_id
        JOIN {DB_CONFIG["database"]}.GAME_SCORES rk ON rk.item_id = r.similar_item_id
        WHERE r.source_item_id = :item_id
        ORDER BY r.similarity_score DESC -- sort by similarity score
        LIMIT :top_n
        """
    )
    result = pd.read_sql(
        query,
        engine,
        params={"item_id": int(item_id), "top_n": int(candidate_pool_size)},
    )
    if result.empty:
        return pd.DataFrame(columns=[
            "source_item_id", "item_id", "item_name", "release_date", "user_reviews", "rating",
            "tags", "developers", "publishers",
            "similarity_score", "pop_score", "quality_score", "age_score"
        ])

    result["source_item_id"] = result["source_item_id"].astype(np.int64)
    result["item_id"] = result["item_id"].astype(np.int64)
    # result["similarity_rank"] = result["similarity_rank"].astype(np.int64)

    # fill all NaNs with 0
    result["similarity_score"] = pd.to_numeric(result["similarity_score"],errors="coerce",).fillna(0.0).astype(np.float32)
    result["pop_score"] = pd.to_numeric(result["pop_score"],errors="coerce",).fillna(0.0).astype(np.float32)
    result["quality_score"] = pd.to_numeric(result["quality_score"],errors="coerce",).fillna(0.0).astype(np.float32)
    result["age_score"] = pd.to_numeric(result["age_score"],errors="coerce",).fillna(0.0).astype(np.float32)
    result["user_reviews"] = pd.to_numeric(result["user_reviews"], errors="coerce").fillna(0).astype(np.int64)

    for col in ["item_name", "release_date", "tags", "developers", "publishers"]:
        result[col] = result[col].fillna("")

    return result



def build_cb_candidates(
    item_id: int,
    engine,
    candidate_pool_size: int = pool_size, # how many similar items to fetch before re-ranking (the more the better chance of good recommendations)
) -> pd.DataFrame:
    """
    Build the base candidate list for content-based recommendations BEFORE applying weights.
    The returned DataFrame already includes the shared re-rank feature columns.
    """
    candidates = fetch_item_similarity_cb(engine=engine, item_id=item_id, candidate_pool_size=candidate_pool_size)
    if candidates.empty:
        return pd.DataFrame(columns=[
            "item_id", "item_name", "release_date", "user_reviews", "rating",
            "tags", "developers", "publishers",
            "similarity_score", "pop_score", "quality_score", "age_score" # for re-ranking
        ])
    return candidates[[
        "item_id", "item_name", "release_date", "user_reviews", "rating",
        "tags", "developers", "publishers",
        "similarity_score", "pop_score", "quality_score", "age_score" # for re-ranking
    ]]


# re-rank the base candidate list and pick top_n games to return
def recommender_cb(
    item_id: int,
    engine,
    top_n: int = top_n, # how many top recommendations to return
    candidate_pool_size: int = pool_size, # how many similar items to fetch before re-ranking (the more the better chance of good recommendations)
    weight_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Content-based recommender using precomputed item-to-item similarity.
    """
    candidates = build_cb_candidates(
        item_id=item_id,
        engine=engine,
        candidate_pool_size=candidate_pool_size,
    )
    if candidates.empty:
        return pd.DataFrame(columns=["item_id", "final_score"])

    # re-rank candidates
    ranked = rerank_candidates(
        candidates_df=candidates,
        base_weight_key="w_cb",
        top_n=top_n,
        weight_overrides=weight_overrides,
    )
    return ranked


# ==================================
# COLLABORATIVE RECOMMENDER
# ==================================


def build_interaction_scores(interactions_df: pd.DataFrame) -> pd.DataFrame:

    """
    Build interaction scores from live user data containing item ownership interactions.

    Expected columns:
        item_id
        hours
        recommendation
        early_access
        median_playtime_forever

    Output columns:
        item_id
        interaction_score
    """

    if interactions_df.empty:
        return pd.DataFrame(columns=["item_id", "interaction_score"])

    df = interactions_df.copy()

    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce")
    df["hours"] = pd.to_numeric(df.get("hours"), errors="coerce").fillna(0.0)
    df["recommendation"] = pd.to_numeric(df.get("recommendation"), errors="coerce").fillna(0)
    df["early_access"] = pd.to_numeric(df.get("early_access"), errors="coerce").fillna(0)
    df["median_playtime_forever"] = pd.to_numeric(df.get("median_playtime_forever"), errors="coerce").fillna(0.0)
    df = df.dropna(subset=["item_id"])


    # IMPORTANT!
    # normalize playtime hours by the game's median lifetime playtime so shorter games are not penalized
    median_hours = np.maximum(df["median_playtime_forever"].fillna(1.0).to_numpy(dtype=np.float32), 1.0)
    normalized_hours = df["hours"].to_numpy(dtype=np.float32) / median_hours

    df["interaction_score"] = (  # calculate interaction score based on recommend and hours
        np.log1p(normalized_hours)  # scale hours relative to the game's typical playtime before logging
        * df["recommendation"].map({2: 1.5, 1: 1.0, 0: 0.5})  # give more weight to recommend=2 and less weight to recommend=0
        * df["early_access"].map({1: 0.8, 0: 1.0}) # give less weight to early access reviews
    ).astype(np.float32)

    df = df[df["interaction_score"] > 0].copy() # exclude meaningless interactions

    if df.empty:
        return pd.DataFrame(columns=["item_id", "interaction_score"])

    df["item_id"] = df["item_id"].astype(np.int64)

    # drop duplicates of item_id, keep the one with highest interaction_score; 
    # there should be no duplicates from live API, but if there is, uncomment this
    # df = (
    #     df.sort_values(["item_id", "interaction_score"], ascending=[True, False])
    #     .drop_duplicates(subset=["item_id"], keep="first")
    #     .reset_index(drop=True)
    # )

    return df[["item_id", "interaction_score"]]


# the math behind the fold-in is based on the ALS optimization objective
# see http://yifanhu.net/PUB/cf.pdf equation (4) and (6)
def _fold_in_user_vector(
    runtime,
    scored_interactions_df: pd.DataFrame,
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
    confidences = (1.0 + als_alpha * observed["interaction_score"].to_numpy(dtype=np.float32)).astype(np.float32)

    # compute user vector via ALS fold-in; see http://yifanhu.net/PUB/cf.pdf equation (4)
    a_matrix = np.array(runtime.item_factor_gram, dtype=np.float32, copy=True)
    a_matrix += als_regularization * np.eye(a_matrix.shape[0], dtype=np.float32)
    b_vector = np.zeros(a_matrix.shape[0], dtype=np.float32)

    for confidence, item_vector in zip(confidences, observed_vectors):
        scaled_confidence = float(confidence - 1.0)
        if scaled_confidence > 0:
            a_matrix += scaled_confidence * np.outer(item_vector, item_vector)
        b_vector += float(confidence) * item_vector

    return np.linalg.solve(a_matrix, b_vector).astype(np.float32), seen_item_ids



# getting games' median playtime from GAME_DATAsince the review dataset does not include it
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

    result["item_id"] = pd.to_numeric(result["item_id"], errors="coerce").astype(np.int64)
    result["median_playtime_forever"] = pd.to_numeric(
        result["median_playtime_forever"],
        errors="coerce",
    ).fillna(0.0).astype(np.float32)
    return result[["item_id", "median_playtime_forever"]]


def fetch_item_similarity_cf(engine, item_id_list: list[int]) -> pd.DataFrame:
    if not item_id_list:
        return pd.DataFrame(columns=["item_id", "item_name", "release_date", "user_reviews", "rating", "tags", "developers", "publishers",
                                     "median_playtime_forever",
                                     "pop_score", "quality_score", "age_score"])

    # query = text(f"""
    #     SELECT *
    #     FROM {DB_CONFIG["database"]}.GAME_SCORES
    #     WHERE item_id IN :item_id_list -- item_ids in the candidate pool before re-ranking
    # """).bindparams(bindparam("item_id_list", expanding=True))

    query = text(
        f"""
        SELECT
            g.item_id,
            g.item_name,
            g.release_date,
            g.user_reviews,
            g.rating,
            g.median_playtime_forever,

            -- handle NULL as string
            COALESCE(g.tags, 'Unknown') AS tags,
            COALESCE(g.developers, 'Unknown') AS developers,
            COALESCE(g.publishers, 'Unknown') AS publishers,

            rk.pop_score,
            rk.quality_score,
            rk.age_score
        FROM {DB_CONFIG["database"]}.GAME_DATA g
        JOIN {DB_CONFIG["database"]}.GAME_SCORES rk ON rk.item_id = g.item_id
        WHERE g.item_id IN :item_id_list -- item_ids in the candidate pool before re-ranking
        LIMIT :top_n
        """
    ).bindparams(bindparam("item_id_list", expanding=True))

    result = pd.read_sql(
        query,
        engine,
        params={
            "item_id_list": [int(x) for x in item_id_list],
            "top_n": int(len(item_id_list)),
        },
    )
    if result.empty:
        return pd.DataFrame(columns=["item_id", "item_name", "release_date", "user_reviews", "rating", "tags", "developers", "publishers",
                                     "median_playtime_forever",
                                     "pop_score", "quality_score", "age_score"])

    result["item_id"] = result["item_id"].astype(np.int32)
    result["median_playtime_forever"] = pd.to_numeric(result["median_playtime_forever"], errors="coerce").fillna(0.0).astype(np.float32)
    result["pop_score"] = pd.to_numeric(result["pop_score"], errors="coerce").fillna(0.0).astype(np.float32)
    result["quality_score"] = pd.to_numeric(result["quality_score"], errors="coerce").fillna(0.0).astype(np.float32)
    result["age_score"] = pd.to_numeric(result["age_score"], errors="coerce").fillna(0.0).astype(np.float32)
    result["user_reviews"] = pd.to_numeric(result["user_reviews"], errors="coerce").fillna(0).astype(np.int32)
    return result[["item_id", "item_name", "release_date", "user_reviews", "rating", "tags", "developers", "publishers",
                                     "median_playtime_forever",
                                     "pop_score", "quality_score", "age_score"]]



# same as build_cb_candidates but for collaborative-filtering
def build_cf_candidates(
    engine,
    runtime,
    live_interactions_df: pd.DataFrame,
    candidate_pool_size: int = pool_size, # how many similar items to fetch before re-ranking (the more the better chance of good recommendations)
) -> pd.DataFrame:
    """
    Build the base candidate list for collaborative-filtering recommendations BEFORE applying weights.
    The returned DataFrame already includes the shared re-rank feature columns.
    """

    interaction_metadata_df = fetch_item_median_playtime(
        engine=engine,
        item_id_list=live_interactions_df["item_id"].dropna().astype(int).tolist(), # list of owned game item_ids fetched from the API
    )
    enriched_interactions_df = live_interactions_df.merge(
        interaction_metadata_df,
        on="item_id",
        how="left",
    )
    scored_interactions_df = build_interaction_scores(enriched_interactions_df) # columns item_id and interaction_score

    if scored_interactions_df.empty:
        return pd.DataFrame(columns=[            
                "item_id", "item_name", "release_date", "user_reviews", "rating",
                "tags", "developers", "publishers",
                "similarity_score", "pop_score", "quality_score", "age_score" # for re-ranking
            ])

    user_vector, seen_item_ids = _fold_in_user_vector(runtime, scored_interactions_df) # fold-in user vector based on interaction scores and item factors
    if user_vector is None:
        return pd.DataFrame(columns=[            
                "item_id", "item_name", "release_date", "user_reviews", "rating",
                "tags", "developers", "publishers",
                "similarity_score", "pop_score", "quality_score", "age_score" # for re-ranking
            ])

    scores = runtime.item_factors @ user_vector # compute CF scores for all items by taking dot product between item factors and user vector
    idx_to_item_id = { # create reverse mapping from ALS item index to item_id
        factor_idx: item_id for item_id, factor_idx in runtime.item_id_to_idx_als.items()
    }
    ordered_item_ids = [idx_to_item_id[idx] for idx in range(len(idx_to_item_id))]
    candidates = pd.DataFrame({"item_id": ordered_item_ids}) # create dataframe of candidate items with their item_ids ordered by ALS index;

    # candidates should be consist only games in the catalog; 
    # from build_review_matrix in recommender_matrices.py
    # runtime_item_ids = set(pd.to_numeric(runtime.items_df["item_id"], errors="coerce").dropna().astype(int).tolist())
    # candidates = candidates[candidates["item_id"].isin(runtime_item_ids)].copy()
    # candidates = candidates.merge(
    #     runtime.items_df[["item_id", "item_name"]],
    #     on="item_id",
    #     how="left",
    # )
    
    candidates["similarity_score"] = scores.astype(np.float32)
    if seen_item_ids:
        candidates = candidates[~candidates["item_id"].isin(seen_item_ids)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=[            
                "item_id", "item_name", "release_date", "user_reviews", "rating",
                "tags", "developers", "publishers",
                "similarity_score", "pop_score", "quality_score", "age_score" # for re-ranking
            ])

    pool_size = max(int(candidate_pool_size), 1)
    candidates = candidates.nlargest(pool_size, "similarity_score").copy() # keep top-N candidates based on CF score before merging with re-rank scores

    rerank_df = fetch_item_similarity_cf( # need to separately fetch rerank df because the candidate pool is initially based on the full item list
        engine=engine,
        item_id_list=candidates["item_id"].astype(int).tolist(),
    )
    candidates = candidates.merge(rerank_df, on="item_id", how="left") # merge with item-level re-rank scores
    return candidates[[
                "item_id", "item_name", "release_date", "user_reviews", "rating",
                "tags", "developers", "publishers",
                "similarity_score", "pop_score", "quality_score", "age_score" # for re-ranking
            ]]


# same as recommend_cb but for collaborative-filtering
def recommend_cf(
    engine,
    live_interactions_df: pd.DataFrame, # dataframe whose data got fetched from live API
    runtime,
    top_n: int = top_n,
    candidate_pool_size: int = pool_size, # how many similar items to fetch before re-ranking (the more the better chance of good recommendations)
    weight_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Collaborative-filtering recommender using ALS fold-in from online user interactions.
    """
    candidates = build_cf_candidates(
        engine,
        live_interactions_df=live_interactions_df,
        runtime=runtime,
        candidate_pool_size=max(candidate_pool_size, top_n),
    )
    if candidates.empty:
        return pd.DataFrame(columns=["item_id", "final_score"])

    ranked = rerank_candidates(
        candidates_df=candidates,
        base_weight_key="w_cf",
        top_n=top_n,
        weight_overrides=weight_overrides,
    )
    return ranked[["item_id", "final_score"]]
