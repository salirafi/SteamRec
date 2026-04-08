
# configurations used for the recommender

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
load_dotenv()



OPENBLAS_NUM_THREADS = 1


DB_CONFIG = {
    "host": os.getenv("STEAM_DB_HOST", "localhost"),
    "port": int(os.getenv("STEAM_DB_PORT", "3306")),
    "user": os.getenv("STEAM_DB_USER"),
    "password": os.getenv("STEAM_DB_PASSWORD"),
}


PATHS = {
    "output_dir": Path("./savefiles/"), # folder storing the pre-computed matrices other than tables
    "prod_ready_dir": Path("./tables/production/"), # folder storing the pre-computed matrices that are ready to be loaded into the database
    "rec_matrices_dir": Path("./tables/rec_matrices/"), # folder storing the pre-computed matrices that are ready to be loaded into the database
}


RECOMMENDER_CONFIG = {
    "w_tags": 1.0, # --> this weight is for the tag similarity score, not intended to be adjustable weights

    "w_age": 0.7,
    "w_anchor_sim": 1.0,
    "w_cf": 0.7,
    "w_popularity": 1.3,
    "w_quality": 1.0,

    # ALS hyperparameters
    "als_factors": 128,
    "als_iterations": 20,
    "als_regularization": 0.1,
    "als_alpha": 1.0,
    "random_state": 42,
}


QUERY_CONFIG = {

    # query for fetching item metadata and features for training
    "item_query": """
        SELECT
            g.item_id,
            g.item_name,
            g.release_date,
            g.positive_ratio,
            g.user_reviews,
            GROUP_CONCAT(DISTINCT t.tag ORDER BY t.tag SEPARATOR ' | ') AS tags
        FROM GAME_DATA g
        LEFT JOIN GAME_TAG t ON g.item_id = t.item_id
        GROUP BY
            g.item_id, g.item_name, g.release_date,
            g.positive_ratio, g.user_reviews
        ORDER BY g.item_id
    """,

    # query for fetching user-item interactions for training
    "interaction_query": """
        SELECT
            r.steam_id,
            r.item_id,
            r.recommend,
            r.hours
            -- r.date
        FROM GAME_REVIEW r
    """,

    "database_query_name": os.getenv("STEAM_DB_QUERY_NAME", "steam_rec_query"),
    "database_prod_name": os.getenv("STEAM_DB_PROD_NAME", "steam_rec"),

}


# SMOKE_TEST_CONFIG = {
#     "steam_id": int(8057089),
#     "top_n_per_anchor": 30,
#     "n_anchor_items": 5,
#     "final_top_n": 30,
# }


def get_db_url() -> str:
    return (
        "mysql+pymysql://"
        f"{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{QUERY_CONFIG['database_prod_name']}"
    )


def get_recommender_config() -> dict[str, Any]:
    return RECOMMENDER_CONFIG.copy()


def qualify_table_name(schema: str, table: str) -> str:
    if not schema:
        return table
    return f"{schema}.{table}"


def get_factor_column_names(n_factors: int | None = None) -> list[str]:
    factor_count = n_factors if n_factors is not None else RECOMMENDER_CONFIG["als_factors"]
    return [f"factor_{idx}" for idx in range(factor_count)]
