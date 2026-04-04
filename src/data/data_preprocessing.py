"""
src/data/data_preprocessing.py

Downloads four Kaggle datasets via kagglehub, normalises labels to [0.0, 1.0],
and merges them into a single unified_training_data.csv with columns:
    text    - raw post / tweet / sentence
    label   - float in [0.0, 1.0]  (1.0 = stressed / high-risk)
    domain  - source dataset tag
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import kagglehub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── output path ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "unified_training_data.csv"


# ── helpers ──────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Basic text normalisation: strip URLs, extra whitespace."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)                       # remove @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                  # strip # from hashtags
    text = re.sub(r"[^\x00-\x7F]+", " ", text)            # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()               # collapse whitespace
    return text


def _find_csv(directory: str, prefer: str | None = None) -> str:
    """Return the first .csv file in *directory*, optionally preferring one that
    contains *prefer* in its name."""
    csvs = list(Path(directory).rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    if prefer:
        matches = [p for p in csvs if prefer.lower() in p.name.lower()]
        if matches:
            return str(matches[0])
    return str(csvs[0])


def _find_csv_by_pattern(directory: str, patterns: list[str]) -> str | None:
    """Search for a CSV matching any of the given name patterns."""
    csvs = list(Path(directory).rglob("*.csv"))
    for pattern in patterns:
        matches = [p for p in csvs if pattern.lower() in p.name.lower()]
        if matches:
            return str(matches[0])
    return None


# ── per-dataset loaders ───────────────────────────────────────────────────────

def load_dreaddit() -> pd.DataFrame:
    """ruchi798/stress-analysis-in-social-media (Dreaddit)
    Original label column: 'label'  (0=no stress, 1=stress)
    """
    logger.info("Downloading ruchi798/stress-analysis-in-social-media …")
    path = kagglehub.dataset_download("ruchi798/stress-analysis-in-social-media")

    csv_file = _find_csv_by_pattern(path, ["dreaddit", "stress"])
    if csv_file is None:
        csv_file = _find_csv(path)

    df = pd.read_csv(csv_file)
    logger.info("Dreaddit raw shape: %s  |  columns: %s", df.shape, list(df.columns))

    # Identify text & label columns
    text_col = next(
        (c for c in df.columns if c.lower() in ("text", "post", "body", "content")),
        None,
    )
    label_col = next(
        (c for c in df.columns if c.lower() in ("label", "stress_label", "target")),
        None,
    )

    if text_col is None or label_col is None:
        raise ValueError(
            f"Cannot identify text/label columns in Dreaddit.  Columns: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": df[label_col].astype(float),
            "domain": "dreaddit",
        }
    )
    return out


def load_behavioural_tweets() -> pd.DataFrame:
    """arshkandroo/behavioural-tweets (Short-form Twitter)
    Label column varies; we map stressed/1/positive → 1.0 else 0.0
    """
    logger.info("Downloading arshkandroo/behavioural-tweets …")
    path = kagglehub.dataset_download("arshkandroo/behavioural-tweets")

    csv_file = _find_csv(path)
    df = pd.read_csv(csv_file)
    logger.info(
        "Behavioural-tweets raw shape: %s  |  columns: %s", df.shape, list(df.columns)
    )

    text_col = next(
        (c for c in df.columns if c.lower() in ("text", "tweet", "content", "message")),
        df.columns[0],
    )

    # Resolve label: look for a stress/sentiment column
    label_col = next(
        (
            c
            for c in df.columns
            if c.lower() in ("label", "stress", "sentiment", "target", "class")
        ),
        None,
    )

    if label_col is None:
        # Fallback: second column if only two columns
        label_col = df.columns[1] if len(df.columns) > 1 else None

    if label_col is None:
        raise ValueError(
            f"Cannot identify label column in behavioural-tweets.  Columns: {list(df.columns)}"
        )

    raw_labels = df[label_col]
    if raw_labels.dtype == object:
        stressed_terms = {"stress", "stressed", "1", "positive", "yes", "high"}
        labels = raw_labels.astype(str).str.lower().isin(stressed_terms).astype(float)
    else:
        # Numeric – normalise to [0, 1]
        mn, mx = float(raw_labels.min()), float(raw_labels.max())
        labels = (raw_labels.astype(float) - mn) / (mx - mn) if mx > mn else raw_labels.astype(float)

    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": labels,
            "domain": "behavioural_tweets",
        }
    )
    return out


def load_suicide_watch() -> pd.DataFrame:
    """nikhileswarkomati/suicide-watch
    Used for Circuit Breaker training; class == 'suicide' → label 1.0
    """
    logger.info("Downloading nikhileswarkomati/suicide-watch …")
    path = kagglehub.dataset_download("nikhileswarkomati/suicide-watch")

    csv_file = _find_csv(path)
    df = pd.read_csv(csv_file)
    logger.info(
        "Suicide-watch raw shape: %s  |  columns: %s", df.shape, list(df.columns)
    )

    text_col = next(
        (c for c in df.columns if c.lower() in ("text", "post", "body", "content")),
        df.columns[0],
    )
    class_col = next(
        (c for c in df.columns if c.lower() in ("class", "label", "category")),
        df.columns[-1],
    )

    labels = (
        df[class_col].astype(str).str.lower().isin({"suicide", "1", "yes"}).astype(float)
    )

    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": labels,
            "domain": "suicide_watch",
        }
    )
    return out


def load_emotions_nlp() -> pd.DataFrame:
    """praveengovi/emotions-dataset-for-nlp
    Classes: joy, sadness, anger, fear, love, surprise
    Stress-adjacent emotions (sadness, anger, fear) → label 1.0, rest → 0.0
    """
    logger.info("Downloading praveengovi/emotions-dataset-for-nlp …")
    path = kagglehub.dataset_download("praveengovi/emotions-dataset-for-nlp")

    csv_file = _find_csv(path, prefer="train")
    df = pd.read_csv(csv_file, names=["text", "emotion"], sep=";", header=None)

    # Some versions already have a header row
    if str(df.iloc[0]["emotion"]).lower() == "emotion":
        df = df.iloc[1:].reset_index(drop=True)

    logger.info(
        "Emotions-NLP raw shape: %s  |  emotion distribution:\n%s",
        df.shape,
        df["emotion"].value_counts(),
    )

    stress_emotions = {"sadness", "anger", "fear"}
    labels = df["emotion"].astype(str).str.lower().isin(stress_emotions).astype(float)

    out = pd.DataFrame(
        {
            "text": df["text"].astype(str),
            "label": labels,
            "domain": "emotions_nlp",
        }
    )
    return out


# ── main pipeline ─────────────────────────────────────────────────────────────

def build_unified_dataset(output_path: Path | None = None) -> pd.DataFrame:
    """Download, clean, and merge all four datasets.

    Returns
    -------
    pd.DataFrame
        Columns: text, label, domain
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    loaders = [
        load_dreaddit,
        load_behavioural_tweets,
        load_suicide_watch,
        load_emotions_nlp,
    ]

    frames: list[pd.DataFrame] = []
    for loader in loaders:
        try:
            df = loader()
            frames.append(df)
            logger.info("Loaded %d rows from %s", len(df), df["domain"].iloc[0])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load dataset from %s: %s", loader.__name__, exc)

    if not frames:
        raise RuntimeError("All dataset downloads failed. Check Kaggle credentials.")

    unified = pd.concat(frames, ignore_index=True)

    # ── cleaning ──────────────────────────────────────────────────────────────
    unified["text"] = unified["text"].apply(_clean_text)
    unified = unified[unified["text"].str.len() > 10].copy()    # drop near-empty rows
    unified = unified.dropna(subset=["text", "label"]).copy()
    unified["label"] = unified["label"].clip(0.0, 1.0)
    unified = unified.drop_duplicates(subset=["text"]).reset_index(drop=True)

    logger.info(
        "Unified dataset: %d rows  |  label distribution:\n%s",
        len(unified),
        unified.groupby("domain")["label"].agg(["count", "mean"]),
    )

    out_path = output_path or OUTPUT_PATH
    unified.to_csv(out_path, index=False)
    logger.info("Saved unified dataset to %s", out_path)

    return unified


if __name__ == "__main__":
    build_unified_dataset()
