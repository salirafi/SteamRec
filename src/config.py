#!usr/bin/env python3
# configurations used for the recommender

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()



DB_CONFIG = {
    "host": os.getenv("STEAM_DB_HOST", "localhost"),
    "port": int(os.getenv("STEAM_DB_PORT", "3306")),
    "user": os.getenv("STEAM_DB_USER"),
    "password": os.getenv("STEAM_DB_PASSWORD"),
    "database": os.getenv("STEAM_DB_NAME", "steam_recommender"),
}

def get_db_url() -> str:
    return (
        "mysql+pymysql://"
        f"{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )



RECOMMENDER_CONFIG = {


    # adjustable weights
    "w_cb": 1.0,
    "w_cf": 0.7,
    "w_age": 0.7,
    "w_popularity": 1.3,
    "w_quality": 1.0,

    # ALS hyperparameters
    "als_factors": 64,
    "als_iterations": 20,
    "als_regularization": 0.1,
    "als_alpha": 1.0,
    "random_state": 42,

    "candidate_pool_size": 140, # how many similar items to fetch from the similarity table for re-ranking (the more the better chance of good recommendations)
    "top_n": 15, # how many top recommendations to return after re-ranking

}


MAP_LABEL = {
    0: "Unknown",
    1: "Overwhelmingly Negative",
    2: "Very Negative",
    3: "Negative",
    4: "Mostly Negative",
    5: "Mixed",
    6: "Mostly Positive",
    7: "Positive",
    8: "Very Positive",
    9: "Overwhelmingly Positive",
}


