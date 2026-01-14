"""Index manager for aggregating and loading multiple filing indexes."""

from pathlib import Path
from typing import Dict, List, Optional

from sec_rag.chunking import Chunk, load_chunks
from sec_rag.config import CHUNKS_DIR, INDEXES_DIR
from sec_rag.embed import Embedder, load_embeddings
from sec_rag.index import SklearnVectorIndex


class IndexManager:
    """Manages aggregated indexes from multiple filings."""
    
    def __init__(self, embedder: Optional[Embedder] = None):
        """
        Initialize index manager.
        
        Args:
            embedder: Optional embedder (will create one if not provided)
        """
        self.embedder = embedder or Embedder()
        self.index: Optional[SklearnVectorIndex] = None
    
    def load_filing_index(
        self,
        ticker: str,
        cik: str,
        form: str,
        year: int,
        accession_nodash: str
    ) -> Optional[SklearnVectorIndex]:
        """
        Load a single filing index from disk.
        
        Args:
            ticker: Ticker symbol
            cik: CIK
            form: Form type
            year: Year
            accession_nodash: Accession number without dashes
            
        Returns:
            SklearnVectorIndex or None if not found
        """
        index_path = INDEXES_DIR / f"{ticker}_{cik}_{form}_{year}_{accession_nodash}.pkl"
        
        if not index_path.exists():
            return None
        
        try:
            return SklearnVectorIndex.load(index_path, self.embedder)
        except Exception as e:
            print(f"Error loading index {index_path}: {e}")
            return None
    
    def aggregate_indexes(
        self,
        tickers: List[str],
        years: List[int],
        form: str = "10-K"
    ) -> SklearnVectorIndex:
        """
        Aggregate indexes from multiple filings.
        
        Args:
            tickers: List of ticker symbols
            years: List of years
            form: Form type
            
        Returns:
            Aggregated SklearnVectorIndex
        """
        from sec_rag.ingest import ticker_to_cik
        
        aggregated_index = SklearnVectorIndex(self.embedder)
        all_chunks: List[Chunk] = []
        
        for ticker in tickers:
            cik = ticker_to_cik(ticker)
            if not cik:
                continue
            
            for year in years:
                # Try to load from index file
                index_files = list(INDEXES_DIR.glob(f"{ticker}_{cik}_{form}_{year}_*.pkl"))
                
                for index_path in index_files:
                    try:
                        filing_index = SklearnVectorIndex.load(index_path, self.embedder)
                        if filing_index and filing_index.chunks:
                            all_chunks.extend(filing_index.chunks)
                    except Exception as e:
                        print(f"Error loading {index_path}: {e}")
                        continue
        
        if all_chunks:
            aggregated_index.add(all_chunks, show_progress=True)
        
        self.index = aggregated_index
        return aggregated_index
    
    def get_index(self) -> SklearnVectorIndex:
        """Get the current index, creating empty one if needed."""
        if self.index is None:
            self.index = SklearnVectorIndex(self.embedder)
        return self.index

