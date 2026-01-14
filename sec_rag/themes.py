"""Risk theme scoring using embedding similarity."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sec_rag.chunking import Chunk
from sec_rag.config import RISK_THEMES
from sec_rag.embed import Embedder
from sec_rag.index import SklearnVectorIndex


def score_theme(
    theme_name: str,
    theme_query: str,
    chunks: List[Chunk],
    embedder: Embedder
) -> float:
    """
    Score a theme by finding max similarity between theme query and chunks.
    
    Args:
        theme_name: Name of the theme
        theme_query: Query text describing the theme
        chunks: List of chunks to score
        index: Vector index
        
    Returns:
        Maximum similarity score (0-1)
    """
    if not chunks:
        return 0.0
    
    # Encode theme query
    query_emb = embedder.encode([theme_query], normalize=True)[0]
    
    # Encode chunks
    chunk_texts = [c.text for c in chunks]
    chunk_embs = embedder.encode(chunk_texts, normalize=True)
    
    # Compute cosine similarities
    similarities = np.dot(chunk_embs, query_emb)
    
    # Return max similarity (or could use mean/top-k mean)
    return float(np.max(similarities))


def score_themes_for_filing(
    chunks: List[Chunk],
    embedder: Embedder,
    themes: Dict[str, str] = RISK_THEMES
) -> Dict[str, float]:
    """
    Score all themes for a filing.
    
    Args:
        chunks: List of chunks from filing
        embedder: Embedder instance
        themes: Dict of theme_name -> theme_query
        
    Returns:
        Dict of theme_name -> score
    """
    scores = {}
    for theme_name, theme_query in themes.items():
        score = score_theme(theme_name, theme_query, chunks, embedder)
        scores[theme_name] = score
    
    return scores


def score_themes_multiple(
    filings_data: Dict[str, Dict[int, Tuple]],
    embedder: Optional[Embedder] = None
) -> pd.DataFrame:
    """
    Score themes across multiple filings.
    
    Args:
        filings_data: Nested dict {ticker: {year: (metadata, chunks, index)}}
        embedder: Optional embedder (will create if not provided)
        
    Returns:
        DataFrame with columns: ticker, year, theme, score
    """
    from sec_rag.embed import Embedder
    
    if embedder is None:
        embedder = Embedder()
    
    rows = []
    
    for ticker, year_data in filings_data.items():
        for year, (metadata, chunks, index) in year_data.items():
            theme_scores = score_themes_for_filing(chunks, embedder)
            
            for theme_name, score in theme_scores.items():
                rows.append({
                    "ticker": ticker,
                    "year": year,
                    "theme": theme_name,
                    "score": score,
                    "filing_date": metadata.filing_date,
                })
    
    df = pd.DataFrame(rows)
    return df

