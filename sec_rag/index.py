"""Vector index for semantic search using scikit-learn."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from sec_rag.chunking import Chunk
from sec_rag.config import INDEXES_DIR
from sec_rag.embed import Embedder
from sec_rag.utils import load_pickle, save_pickle


class SklearnVectorIndex:
    """
    Cosine similarity search using scikit-learn NearestNeighbors.
    Suitable for up to tens of thousands of chunks.
    """
    
    def __init__(self, embedder: Embedder):
        """
        Initialize vector index.
        
        Args:
            embedder: Embedder instance
        """
        self.embedder = embedder
        self.embeddings: np.ndarray | None = None
        self.nn: NearestNeighbors | None = None
        self.chunks: List[Chunk] = []
    
    def add(self, chunks: List[Chunk], show_progress: bool = True) -> None:
        """
        Add chunks to the index.
        
        Args:
            chunks: List of Chunk objects to add
            show_progress: Whether to show progress bar
        """
        if not chunks:
            return
        
        # Encode chunks
        new_embeddings = self.embedder.encode_chunks(chunks, show_progress=show_progress)
        new_embeddings = np.asarray(new_embeddings, dtype=np.float32)
        
        # Combine with existing embeddings
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Add chunks
        self.chunks.extend(chunks)
        
        # Rebuild index
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1)
        self.nn.fit(self.embeddings)
    
    def search(
        self,
        query: str,
        top_k: int = 6,
        filter_meta: dict | None = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filter_meta: Optional metadata filter (e.g., {"ticker": "AAPL"})
            
        Returns:
            List of (Chunk, similarity_score) tuples, sorted by similarity
            
        Raises:
            RuntimeError: If index is empty
        """
        if self.nn is None or self.embeddings is None or len(self.chunks) == 0:
            raise RuntimeError("Index is empty. Add chunks first.")
        
        # Encode query
        q_emb = self.embedder.encode([query], normalize=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        
        # Search
        k = min(top_k * 3, len(self.chunks))  # Get more candidates for filtering
        distances, indices = self.nn.kneighbors(q_emb, n_neighbors=k)
        sims = 1.0 - distances[0]  # Convert distance to similarity
        
        # Build results
        results: List[Tuple[Chunk, float]] = []
        for idx, sim in zip(indices[0], sims):
            chunk = self.chunks[int(idx)]
            
            # Apply metadata filter if provided
            if filter_meta:
                match = True
                for key, value in filter_meta.items():
                    if chunk.meta.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            results.append((chunk, float(sim)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, filepath: Path) -> None:
        """Save index to disk."""
        data = {
            "embeddings": self.embeddings,
            "chunks": self.chunks,
            "model_name": self.embedder.model_name
        }
        save_pickle(data, filepath)
    
    @classmethod
    def load(cls, filepath: Path, embedder: Embedder) -> "SklearnVectorIndex":
        """Load index from disk."""
        data = load_pickle(filepath)
        index = cls(embedder)
        index.embeddings = data["embeddings"]
        index.chunks = data["chunks"]
        index.nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1)
        index.nn.fit(index.embeddings)
        return index

