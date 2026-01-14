"""Utility functions."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def save_json(data: Any, filepath: Path) -> None:
    """Save data as JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> Any:
    """Load data from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data: Any, filepath: Path) -> None:
    """Save data as pickle."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath: Path) -> Any:
    """Load data from pickle."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_numpy(array: np.ndarray, filepath: Path) -> None:
    """Save numpy array."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, array)


def load_numpy(filepath: Path) -> np.ndarray:
    """Load numpy array."""
    return np.load(filepath)


def cik_to_ticker(cik: str) -> str:
    """Convert CIK to ticker (placeholder - would need a mapping)."""
    # In a real implementation, you'd have a CIK->ticker mapping
    return cik


def ticker_to_cik(ticker: str) -> Optional[str]:
    """Convert ticker to CIK (placeholder - would need a mapping)."""
    # In a real implementation, you'd have a ticker->CIK mapping
    # For now, we'll assume CIK is provided directly or use a simple lookup
    return None


def normalize_cik(cik: str) -> str:
    """Normalize CIK to 10-digit zero-padded string."""
    return cik.zfill(10)


def get_filing_cache_key(ticker: str, cik: str, form: str, year: int, accession: str) -> str:
    """Generate a cache key for a filing."""
    return f"{ticker}_{cik}_{form}_{year}_{accession}"


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)

