"""Embedding generation using SentenceTransformers."""

from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from sec_rag.chunking import Chunk
from sec_rag.config import EMBED_MODEL_NAME, EMBEDDINGS_DIR
from sec_rag.utils import load_numpy, save_numpy


class Embedder:
    """Wrapper for SentenceTransformer embedding model."""
    
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        """
        Initialize embedder.
        
        Args:
            model_name: SentenceTransformer model name
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            normalize: Whether to normalize embeddings (for cosine similarity)
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (n_texts, dim)
        """
        return self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    def encode_chunks(self, chunks: List[Chunk], show_progress: bool = True) -> np.ndarray:
        """
        Encode chunks to embeddings.
        
        Args:
            chunks: List of Chunk objects
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        texts = [c.text for c in chunks]
        return self.encode(texts, normalize=True, show_progress=show_progress)


def save_embeddings(embeddings: np.ndarray, filepath: Path) -> None:
    """Save embeddings to disk."""
    save_numpy(embeddings, filepath)


def load_embeddings(filepath: Path) -> np.ndarray:
    """Load embeddings from disk."""
    return load_numpy(filepath)

