from pathlib import Path
from dataclasses import dataclass
import ast
import re
import os
import time
import uuid
from threading import Lock

import numpy as np
from sqlalchemy import create_engine
from flask import Flask, request, jsonify, render_template
from src._get_steam_API import fetch_owned_games_df
from src.config import get_db_url, PATHS
from src.recommender import (
    build_cb_candidates,
    build_cf_candidates,
    load_runtime,
    rerank_candidates,
)
import src.recommender as recommender_service

app = Flask(__name__)

OUTPUT_DIR = PATHS["output_dir"]
DEFAULT_TOP_N = 15 # number of recommendations to return; adjust based on UI design and performance considerations
DEFAULT_CB_POOL_SIZE = 100 # number of similar games to fetch from the database for re-ranking in game-based search; adjust based on expected performance and recommendation quality tradeoff
CACHE_TTL_SECONDS = 30 * 60 # 30 minutes TTL for cached search results; adjust based on expected user behavior and performance needs
MAX_CACHED_SEARCHES = 32 # to prevent memory bloat; adjust based on expected traffic and memory constraints


###### !!! IMPORTANT: ############

# the web app will load the runtime from disk on startup and used it throughout the session
# for the current game database which only contains ~50k games, this would still be manageable and fast enough
# but if the game database grows significantly, a more dynamic loading strategy for the runtime would be necessary
runtime = load_runtime(OUTPUT_DIR)

###### !!! IMPORTANT: ############




engine = None
_search_cache = {}
_search_cache_lock = Lock()


@dataclass
class CachedSearch:
    search_id: str
    mode: str
    base_score_column: str
    base_weight_key: str
    candidates_df: object
    created_at: float
    context: dict

@app.get("/")
def serve_index():
    return render_template("steam_recommender.html")


# def get_engine():
#     global engine
#     if engine is None:
#         host = os.getenv("STEAM_DB_HOST", "localhost")
#         port = int(os.getenv("STEAM_DB_PORT", "3306"))
#         user = os.getenv("STEAM_DB_USER", "root")
#         password = os.getenv("STEAM_DB_PASSWORD", "Keppler172b!")
#         database = os.getenv("STEAM_DB_NAME", "steam_prod")
#         engine = create_engine(
#             f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
#         )
#     return engine


# cleanup expired cache entries based on TTL and limit total number of cached searches to prevent memory bloat
# this is called before accessing the cache to ensure stale data not returned and to keep memory usage in check
def _cleanup_search_cache(now=None):
    current_time = now if now is not None else time.time()
    expired_keys = [
        cache_key
        for cache_key, cached in _search_cache.items()
        if current_time - cached.created_at > CACHE_TTL_SECONDS
    ]
    for cache_key in expired_keys:
        _search_cache.pop(cache_key, None)

    while len(_search_cache) > MAX_CACHED_SEARCHES:
        oldest_key = min(
            _search_cache,
            key=lambda cache_key: _search_cache[cache_key].created_at,
        )
        _search_cache.pop(oldest_key, None)

# store candidate df and search context in cache with a unique search_id
def _store_cached_search(
    mode,
    candidates_df,
    base_score_column,
    base_weight_key,
    context,
):
    search_id = uuid.uuid4().hex
    cached = CachedSearch(
        search_id=search_id,
        mode=mode,
        base_score_column=base_score_column,
        base_weight_key=base_weight_key,
        candidates_df=candidates_df.copy(),
        created_at=time.time(),
        context=dict(context),
    )
    with _search_cache_lock:
        _cleanup_search_cache(now=cached.created_at)
        _search_cache[search_id] = cached
    return cached


def _get_cached_search(search_id):
    with _search_cache_lock:
        _cleanup_search_cache()
        cached = _search_cache.get(search_id)
        if cached is None:
            return None
        cached.created_at = time.time()
        return cached


def _parse_weight_value(raw_value, default_value):
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return float(default_value)


def _build_weight_overrides(mode, weights):
    overrides = {
        "w_popularity": _parse_weight_value(
            weights.get("popularity"),
            recommender_service.config["w_popularity"],
        ),
        "w_quality": _parse_weight_value(
            weights.get("quality"),
            recommender_service.config["w_quality"],
        ),
        "w_age": _parse_weight_value(
            weights.get("age"),
            recommender_service.config["w_age"],
        ),
    }
    similarity_weight = _parse_weight_value(
        weights.get("similarity"),
        recommender_service.config["w_anchor_sim"],
    )
    if mode == "user":
        overrides["w_cf"] = similarity_weight
    else:
        overrides["w_anchor_sim"] = similarity_weight
    return overrides



def rating_label(x):
    if x is None:
        return "Unknown"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "Unknown"

    if x == 9:
        return "Overwhelmingly Positive"
    if x == 8:
        return "Very Positive"
    if x == 7:
        return "Mostly Positive"
    if x == 6:
        return "Positive"
    if x == 5:
        return "Mixed"
    if x == 4:
        return "Negative"
    if x == 3:
        return "Mostly Negative"
    if x == 2:
        return "Very Negative"
    if x == 1:
        return "Overwhelmingly Negative"
    return "Unknown"

def split_pipe_values(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]

    if isinstance(value, tuple):
        return [str(part).strip() for part in value if str(part).strip()]

    if isinstance(value, np.ndarray):
        return [str(part).strip() for part in value.tolist() if str(part).strip()]

    text = str(value).strip()
    if not text:
        return []

    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(part).strip() for part in parsed if str(part).strip()]
        except (ValueError, SyntaxError):
            pass

        # handles numpy-style string arrays like ['2D' 'Base Building' 'City Builder']
        quoted_parts = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
        flattened = [part.strip() for pair in quoted_parts for part in pair if part.strip()]
        if flattened:
            return flattened

    return [text]


# lists of selected recommended games
def build_recommendation_payload(recs):
    items_meta = runtime.items_df[
        ["item_id", "item_name", "release_date", "user_reviews", "rating", 
         "tags", 
        # "genres", 
        # "developers", "publishers"
         ]
    ].copy()

    recs = recs.merge(items_meta, on="item_id", how="left")
    recs["rating"] = recs["rating"].apply(rating_label)
    recs["steamUrl"] = recs["item_id"].apply(
        lambda x: f"https://store.steampowered.com/app/{int(x)}/"
    )
    recs["image"] = recs["item_id"].apply(
        lambda x: f"https://shared.cloudflare.steamstatic.com/store_item_assets/steam/apps/{int(x)}/header.jpg"
    )
    recs["id"] = recs["item_id"]
    recs["name"] = recs["item_name"]
    recs["releaseDate"] = recs["release_date"].fillna("Unknown")
    recs["recommendationScore"] = recs["final_score"]
    recs["userReviews"] = recs["user_reviews"].fillna(0)

    recs["tags"] = recs["tags"].apply(split_pipe_values)
    # recs["genres"] = recs["genres"].apply(split_pipe_values)
    # recs["developers"] = recs["developers"].apply(split_pipe_values)
    # recs["publishers"] = recs["publishers"].apply(split_pipe_values)
    
    # recs["tags"] = [["Action", "Adventure"].copy() for _ in range(len(recs))]
    recs["genres"] = None
    recs["developers"] = [["Unknown Developer"] for _ in range(len(recs))]
    recs["publishers"] = [["Unknown Publisher"] for _ in range(len(recs))]


    recs = recs.reset_index(drop=True)
    recs["rank"] = recs.index + 1

    payload = []
    for row in recs.itertuples(index=False):
        payload.append({
            "rank": int(row.rank),
            "id": int(row.id),
            "name": str(row.name),
            "image": str(row.image),
            "steamUrl": str(row.steamUrl),
            "releaseDate": str(row.releaseDate),
            "userReviews": int(row.userReviews),
            "rating": str(row.rating),
            "recommendationScore": float(row.recommendationScore),
            "tags": row.tags,
            "genres": row.genres,
            "developers": row.developers,
            "publishers": row.publishers,
        })
    return payload


def _build_ranked_response(cached_search, weights, top_n=DEFAULT_TOP_N, message=""):
    recs = rerank_candidates(
        candidates_df=cached_search.candidates_df,
        base_score_column=cached_search.base_score_column,
        base_weight_key=cached_search.base_weight_key,
        top_n=top_n,
        weight_overrides=_build_weight_overrides(cached_search.mode, weights),
    )
    return jsonify({
        "results": build_recommendation_payload(recs),
        "message": message,
        "search_id": cached_search.search_id,
        "context": cached_search.context,
    })


def build_game_suggestion_payload(rows):
    payload = []
    for row in rows.itertuples(index=False):
        payload.append({
            "id": int(row.item_id),
            "name": str(row.item_name),
            "image": f"https://shared.cloudflare.steamstatic.com/store_item_assets/steam/apps/{int(row.item_id)}/header.jpg",
        })
    return payload


# ==========================
# Catalog search endpoint
# ==========================

# normalize typed game name
def normalize_name(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())

# this function is used to build candidate dataframe for game-based search
def search_game_catalog(query, limit=8):
    normalized_query = normalize_name(query)
    if not normalized_query:
        return runtime.items_df.iloc[0:0][["item_id", "item_name"]]

    catalog = runtime.items_df[["item_id", "item_name"]].dropna().copy()
    catalog["normalized_name"] = catalog["item_name"].map(normalize_name)

    # matching strategy:
    starts_with = catalog["normalized_name"].str.startswith(normalized_query, na=False) # strict prefix match
    contains = catalog["normalized_name"].str.contains(re.escape(normalized_query), na=False) # looser substring match
    token_match = catalog["normalized_name"].map(
        lambda name: all(token in name for token in normalized_query.split()) # token-based match; only matches if all tokens in the query are present in the name
    )

    matches = catalog[starts_with | contains | token_match].copy() # combine all matched candidates, with potential duplicates if a name matches multiple criteria
    if matches.empty:
        return matches[["item_id", "item_name"]]

    matches["match_rank"] = ( # rank: strict prefix match > token-based match > loose substring match
        (~starts_with.loc[matches.index]).astype(int) * 2 +
        (~contains.loc[matches.index]).astype(int)
    )
    matches["name_len"] = matches["normalized_name"].str.len() # secondary sort by name length to prioritize shorter names when match_rank is the same
    matches = matches.sort_values(
        ["match_rank", "name_len", "item_name", "item_id"],
        ascending=[True, True, True, True]
    )
    return matches.head(limit)[["item_id", "item_name"]]

@app.get("/api/search/games")
def search_games():
    query = request.args.get("q", "", type=str)
    matches = search_game_catalog(query, limit=8)
    return jsonify({"results": build_game_suggestion_payload(matches)})


# ==========================
# Content-based recommendation endpoint
# ==========================


@app.post("/api/recommend/game")
def recommend_game():
    try:
        data = request.get_json(force=True)
        item_id = int(data["item_id"])
        weights = data.get("weights", {})
    except (TypeError, ValueError, KeyError):
        return jsonify({"results": [], "message": "Invalid game selection."}), 400
    try:
        candidates = build_cb_candidates(
            item_id=item_id,
            engine=create_engine(get_db_url()),
            similarity_pool_size=DEFAULT_CB_POOL_SIZE,
        )
    except Exception as exc:
        return jsonify({"results": [], "message": f"Failed to load similar games: {exc}"}), 500

    if candidates.empty:
        return jsonify({"results": [], "message": "No similar games found for this title."})


    # candidate games BEFORE re-ranking are stored in cache to be used for re-ranking when user adjusts weights
    cached_search = _store_cached_search(
        mode="game",
        candidates_df=candidates,
        base_score_column="similarity_score",
        base_weight_key="w_anchor_sim",
        context={"item_id": item_id},
    )
    return _build_ranked_response(cached_search, weights)


# ==========================
# Collaborative-filtering recommendation endpoint
# ==========================


@app.post("/api/recommend/user")
def recommend_user():
    try:
        data = request.get_json(force=True)
        steam_id = str(data["steam_id"]).strip()
        weights = data.get("weights", {})
    except (TypeError, ValueError, KeyError):
        return jsonify({"results": [], "message": "Invalid Steam user ID."}), 400

    try:
        interactions_df = fetch_owned_games_df(steam_id)
    except Exception as exc:
        return jsonify({"results": [], "message": "Failed to load Steam library,"}), 500

    if interactions_df.empty:
        return jsonify({"results": [], "message": f"No owned games found for {steam_id}."})

    candidates = build_cf_candidates(
        live_interactions_df=interactions_df,
        runtime=runtime,
    )
    if candidates.empty:
        return jsonify({
            "results": [],
            "message": "No recommendations could be built from this Steam library.",
        })

    cached_search = _store_cached_search(
        mode="user",
        candidates_df=candidates,
        base_score_column="cf_score",
        base_weight_key="w_cf",
        context={"steam_id": steam_id},
    )
    return _build_ranked_response(cached_search, weights)


@app.post("/api/recommend/rerank")
def rerank_search():
    try:
        data = request.get_json(force=True)
        search_id = str(data["search_id"]).strip()
        weights = data.get("weights", {})
    except (TypeError, ValueError, KeyError):
        return jsonify({"results": [], "message": "Invalid rerank request."}), 400

    cached_search = _get_cached_search(search_id)
    if cached_search is None:
        return jsonify({
            "results": [],
            "message": "This recommendation session expired. Please run the search again.",
        }), 404

    return _build_ranked_response(cached_search, weights)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
