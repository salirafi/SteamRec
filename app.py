#!/usr/bin/env python3

from dataclasses import dataclass
import re
import time
import uuid
from threading import Lock

import pandas as pd
from sqlalchemy import create_engine, text
from flask import Flask, request, jsonify, render_template
from src._get_steam_API import fetch_owned_games_df
from src.config import DB_CONFIG, get_db_url, RECOMMENDER_CONFIG, MAP_LABEL
from src.helpers import split_pipe_values, load_runtime
from src.recommender import (
    build_cb_candidates,
    build_cf_candidates,
    rerank_candidates,
)

app = Flask(__name__)

DEFAULT_TOP_N = RECOMMENDER_CONFIG["top_n"] # number of recommendations to return; adjust based on UI design and performance considerations
DEFAULT_CB_POOL_SIZE = RECOMMENDER_CONFIG["candidate_pool_size"] # number of similar games to fetch from the database for re-ranking in game-based search; adjust based on expected performance and recommendation quality tradeoff
CACHE_TTL_SECONDS = 30 * 60 # 30 minutes TTL for cached search results; adjust based on expected user behavior and performance needs
MAX_CACHED_SEARCHES = 32 # to prevent memory bloat; adjust based on expected traffic and memory constraints


# =============== IMPORTANT! ===================

# the web app will load the runtime from disk on startup and used it throughout the session
# for the current game database which only contains ~50k games, this would still be manageable and fast enough
# but if the game database grows significantly, a more dynamic loading strategy for the runtime would be necessary
runtime = load_runtime()

# ==============================================


engine = create_engine(get_db_url())
_search_cache = {}
_search_cache_lock = Lock()


@dataclass
class CachedSearch:
    search_id: str
    mode: str
    base_weight_key: str
    candidates_df: object
    created_at: float
    context: dict


@app.get("/")
def serve_index():
    return render_template("steam_recommender.html")




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
    # base_score_column,
    base_weight_key,
    context,
):
    search_id = uuid.uuid4().hex
    cached = CachedSearch(
        search_id=search_id,
        mode=mode,
        # base_score_column=base_score_column,
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
            RECOMMENDER_CONFIG["w_popularity"],
        ),
        "w_quality": _parse_weight_value(
            weights.get("quality"),
            RECOMMENDER_CONFIG["w_quality"],
        ),
        "w_age": _parse_weight_value(
            weights.get("age"),
            RECOMMENDER_CONFIG["w_age"],
        ),
    }
    similarity_weight = _parse_weight_value(
        weights.get("similarity"),
        RECOMMENDER_CONFIG["w_cb"],
    )
    if mode == "user":
        overrides["w_cf"] = similarity_weight
    else:
        overrides["w_cb"] = similarity_weight
    return overrides




# lists of selected recommended games
def build_recommendation_payload(recs):

    recs["rating"] = recs["rating"].map(MAP_LABEL).fillna("Unknown")
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
    recs["developers"] = recs["developers"].apply(split_pipe_values)
    recs["publishers"] = recs["publishers"].apply(split_pipe_values)


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
            "developers": row.developers,
            "publishers": row.publishers,
        })
    return payload


def _build_ranked_response(cached_search, weights, top_n=DEFAULT_TOP_N, message=""):
    recs = rerank_candidates(
        candidates_df=cached_search.candidates_df,
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


# build candidate dataframe for game-based search
def search_game_catalog(query, limit=5):

    normalized_query = re.sub(r"\s+", " ", str(query or "").strip().lower()) # normalize typed game name
    if not normalized_query:
        return pd.DataFrame(columns=["item_id", "item_name"]) # return empty dataframe if query is empty


    # build SQL query with multiple matching strategies: prefix match, token match, and substring match; and order results by relevance

    token_clauses = " AND ".join(
        [f"LOWER(item_name) LIKE :token_{idx}" for idx, _ in enumerate(normalized_query.split())]
    )
    if not token_clauses:
        token_clauses = "1 = 1"

    query_sql = text(
        f"""
        SELECT
            item_id,
            item_name
        FROM {DB_CONFIG["database"]}.GAME_DATA
        WHERE
            LOWER(item_name) LIKE :prefix_query
            OR LOWER(item_name) LIKE :like_query
            OR ({token_clauses})
        ORDER BY
            CASE
                WHEN LOWER(item_name) LIKE :prefix_query THEN 0
                WHEN ({token_clauses}) THEN 1
                ELSE 2
            END,
            CHAR_LENGTH(item_name),
            item_name,
            item_id
        LIMIT :limit
        """
    )
    params = {
        "prefix_query": f"{normalized_query}%",
        "like_query": f"%{normalized_query}%",
        "limit": int(limit),
    }
    for idx, token in enumerate(normalized_query.split()):
        params[f"token_{idx}"] = f"%{token}%"
    return pd.read_sql(query_sql, engine, params=params)


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
            engine=engine,
            candidate_pool_size=DEFAULT_CB_POOL_SIZE,
        )
    except Exception as exc:
        return jsonify({"results": [], "message": f"Failed to load similar games: {exc}"}), 500

    if candidates.empty:
        return jsonify({"results": [], "message": "No similar games found for this title."})

    # candidate games BEFORE re-ranking are stored in cache to be used for re-ranking when user adjusts weights
    cached_search = _store_cached_search(
        mode="game",
        candidates_df=candidates,
        base_weight_key="w_cb",
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
        engine=engine,
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
        base_weight_key="w_cf",
        context={"steam_id": steam_id},
    )
    return _build_ranked_response(cached_search, weights)


# ==========================
# Re-reranking based candidates
# ==========================


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



# for cron job
@app.route("/ping")
def ping():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(debug=True, port=8000)
