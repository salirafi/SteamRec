import pandas as pd
from pathlib import Path
import numpy as np
import csv
import ast
import re
import json
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_GAME_FACTORS_DIR = BASE_DIR / "tables" / "production"



# ==================================
# HELPERS FOR PRE_PROCESSING STAGES
# ==================================


# from https://www.reddit.com/r/Steam/comments/ivz45n/what_does_the_steam_ratings_like_very_negative_or/
def rating_label(positive_ratio: float, user_reviews: int) -> int:
    if positive_ratio is None or user_reviews is None:
        return 0
    try:
        positive_ratio = float(positive_ratio)
        user_reviews = int(user_reviews)
    except (TypeError, ValueError):
        return 0

    if positive_ratio > 95.0 and user_reviews > 500:
        return 9
    if positive_ratio > 80.0 and user_reviews > 50:
        return 8
    if positive_ratio > 80.0 and user_reviews > 10:
        return 7
    if positive_ratio > 70.0 and user_reviews > 10:
        return 6
    if positive_ratio > 40.0 and user_reviews > 10:
        return 5
    if positive_ratio > 20.0 and user_reviews > 10:
        return 4
    if positive_ratio > 0.0 and user_reviews > 500:
        return 1
    if positive_ratio > 0.0 and user_reviews > 50:
        return 2
    if positive_ratio > 0.0 and user_reviews > 10:
        return 3
    return 0



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
        quoting=csv.QUOTE_ALL, # quote all fields,
        float_format="%.2f",
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





# ==================================
# HELPERS FOR RECOMMENDER ANALYSIS
# ==================================


@dataclass
class RuntimeArtifacts:
    item_factors: np.ndarray
    item_id_to_idx_als: dict[int, int]
    item_factor_gram: np.ndarray

_RUNTIME_CACHE: dict[Path, RuntimeArtifacts] = {}


# load runtime item ALS factors
def load_runtime() -> RuntimeArtifacts:
    cached = _RUNTIME_CACHE.get(OUTPUT_GAME_FACTORS_DIR)
    if cached is not None:
        return cached

    item_factors = np.load(OUTPUT_GAME_FACTORS_DIR / "item_factors.npy").astype(np.float32)

    raw = json.loads((OUTPUT_GAME_FACTORS_DIR / "item_id_to_idx_als.json").read_text(encoding="utf-8"))
    item_id_to_idx_als = {int(key): int(value) for key, value in raw.items()}

    # compute item factor gram matrix for fold-in
    item_factor_gram = np.asarray(item_factors.T @ item_factors, dtype=np.float32)

    runtime = RuntimeArtifacts(
        item_factors=item_factors,
        item_id_to_idx_als=item_id_to_idx_als,
        item_factor_gram=item_factor_gram,
    )
    _RUNTIME_CACHE[OUTPUT_GAME_FACTORS_DIR] = runtime
    return runtime



def _safe_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    smin = s.min()
    smax = s.max()
    if smax <= smin:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - smin) / (smax - smin)


# split pipe-separated values into lists, 
# also handles some edge cases
def split_pipe_values(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]

    if isinstance(value, tuple):
        return [str(part).strip() for part in value if str(part).strip()]

    if isinstance(value, np.ndarray):
        return [str(part).strip() for part in value.tolist() if str(part).strip()]

    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text or text in {"<NA>", "nan", "None"}:
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
