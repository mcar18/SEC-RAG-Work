"""Filing ingestion pipeline."""

from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from sec_rag.chunking import Chunk, create_chunks_from_text, save_chunks
from sec_rag.config import (
    CHUNKS_DIR, DEFAULT_FORM_TYPE, EMBEDDINGS_DIR, INDEXES_DIR, PARSED_DIR
)
from sec_rag.embed import Embedder, save_embeddings
from sec_rag.filings import FilingMetadata, fetch_filing_metadata
from sec_rag.index import SklearnVectorIndex
from sec_rag.parse import download_and_parse_filing
from sec_rag.utils import get_filing_cache_key, normalize_cik


# Simple ticker to CIK mapping for common companies
# In production, you'd fetch this from SEC or use a proper mapping service
# Top 20 largest US stocks by market cap (as of 2024)
TICKER_TO_CIK: Dict[str, str] = {
    # Technology
    "AAPL": "0000320193",  # Apple Inc.
    "MSFT": "0000789019",  # Microsoft Corporation
    "NVDA": "0001045810",  # NVIDIA Corporation
    "GOOGL": "0001652044",  # Alphabet Inc. (Class A)
    "GOOG": "0001652044",   # Alphabet Inc. (Class C) - same CIK
    "AMZN": "0001018724",  # Amazon.com Inc.
    "META": "0001326801",  # Meta Platforms Inc.
    "TSLA": "0001318605",  # Tesla, Inc.
    "AVGO": "0001730168",  # Broadcom Inc.
    "ORCL": "0001341439",  # Oracle Corporation
    "ADBE": "0000796343",  # Adobe Inc.
    # Financial Services
    "JPM": "0000019617",   # JPMorgan Chase & Co.
    "V": "0001403161",     # Visa Inc.
    "MA": "0001141391",    # Mastercard Incorporated
    "BAC": "0000010734",   # Bank of America Corp
    # Healthcare
    "JNJ": "0000200406",   # Johnson & Johnson
    "UNH": "0000731766",   # UnitedHealth Group Inc.
    "LLY": "0000059478",   # Eli Lilly and Company
    # Consumer
    "WMT": "0000104169",   # Walmart Inc.
    "PG": "0000080424",    # Procter & Gamble Company
    "HD": "0000354950",    # The Home Depot, Inc.
    "COST": "0000909832",  # Costco Wholesale Corporation
    "DIS": "0001001039",   # The Walt Disney Company
    # Energy
    "XOM": "0000034088",   # Exxon Mobil Corporation
    # Conglomerate
    "BRK.B": "0001067983", # Berkshire Hathaway Inc. (Class B)
    "BRKB": "0001067983",  # Alternative format
}


def ticker_to_cik(ticker: str) -> Optional[str]:
    """
    Convert ticker to CIK.
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        CIK string or None if not found
    """
    ticker_upper = ticker.upper()
    return TICKER_TO_CIK.get(ticker_upper)


def ingest_filing(
    ticker: str,
    cik: Optional[str] = None,
    form: str = DEFAULT_FORM_TYPE,
    year: Optional[int] = None,
    use_cache: bool = True
) -> tuple[FilingMetadata, List[Chunk], SklearnVectorIndex]:
    """
    Ingest a single filing: download, parse, chunk, embed, and index.
    
    Args:
        ticker: Company ticker symbol
        cik: Company CIK (if None, will try to look up from ticker)
        form: Form type (default: "10-K")
        year: Optional year filter
        use_cache: Whether to use cached files
        
    Returns:
        Tuple of (metadata, chunks, index)
    """
    # Resolve CIK
    if not cik:
        cik = ticker_to_cik(ticker)
        if not cik:
            raise ValueError(f"Could not resolve CIK for ticker {ticker}. Please provide CIK directly.")
    
    cik = normalize_cik(cik)
    
    # Fetch metadata
    print(f"Fetching metadata for {ticker} (CIK: {cik}) {form}...")
    metadata = fetch_filing_metadata(ticker, cik, form=form, year=year)
    print(f"Found filing: {metadata.filing_date} ({metadata.accession_nodash})")
    
    # Download and parse
    print(f"Downloading and parsing filing...")
    text, resolved_doc = download_and_parse_filing(metadata, use_cache=use_cache)
    metadata.resolved_document = resolved_doc
    print(f"Extracted {len(text):,} characters")
    
    # Chunk
    print(f"Chunking text...")
    chunk_metadata = {
        "ticker": ticker,
        "cik": cik,
        "form": form,
        "accession_number": metadata.accession_number,
        "accession_nodash": metadata.accession_nodash,
        "filing_date": metadata.filing_date,
        "year": metadata.year,
        "primary_document": metadata.primary_document,
        "resolved_document": metadata.resolved_document,
    }
    chunks = create_chunks_from_text(text, chunk_metadata)
    print(f"Created {len(chunks)} chunks")
    
    # Save chunks
    if use_cache:
        chunks_path = CHUNKS_DIR / f"{ticker}_{cik}_{form}_{metadata.year}_{metadata.accession_nodash}.json"
        save_chunks(chunks, chunks_path)
    
    # Embed
    print(f"Generating embeddings...")
    embedder = Embedder()
    embeddings = embedder.encode_chunks(chunks, show_progress=True)
    
    # Save embeddings
    if use_cache:
        emb_path = EMBEDDINGS_DIR / f"{ticker}_{cik}_{form}_{metadata.year}_{metadata.accession_nodash}.npy"
        save_embeddings(embeddings, emb_path)
    
    # Build index
    print(f"Building vector index...")
    index = SklearnVectorIndex(embedder)
    index.add(chunks, show_progress=False)
    
    # Save index
    if use_cache:
        index_path = INDEXES_DIR / f"{ticker}_{cik}_{form}_{metadata.year}_{metadata.accession_nodash}.pkl"
        index.save(index_path)
    
    return metadata, chunks, index


def ingest_multiple(
    tickers: List[str],
    years: List[int],
    form: str = DEFAULT_FORM_TYPE,
    cik_map: Optional[Dict[str, str]] = None,
    use_cache: bool = True
) -> Dict[str, Dict[int, tuple[FilingMetadata, List[Chunk], SklearnVectorIndex]]]:
    """
    Ingest multiple filings across tickers and years.
    
    Args:
        tickers: List of ticker symbols
        years: List of years to fetch
        form: Form type
        cik_map: Optional dict mapping ticker -> CIK
        use_cache: Whether to use cached files
        
    Returns:
        Nested dict: {ticker: {year: (metadata, chunks, index)}}
    """
    if cik_map is None:
        cik_map = {}
    
    results: Dict[str, Dict[int, tuple]] = {}
    
    for ticker in tickers:
        cik = cik_map.get(ticker) or ticker_to_cik(ticker)
        if not cik:
            print(f"Warning: Could not resolve CIK for {ticker}, skipping...")
            continue
        
        results[ticker] = {}
        
        for year in years:
            try:
                print(f"\n{'='*60}")
                print(f"Ingesting {ticker} {form} for year {year}")
                print(f"{'='*60}")
                
                metadata, chunks, index = ingest_filing(
                    ticker=ticker,
                    cik=cik,
                    form=form,
                    year=year,
                    use_cache=use_cache
                )
                
                results[ticker][year] = (metadata, chunks, index)
                
            except Exception as e:
                print(f"Error ingesting {ticker} {year}: {e}")
                continue
    
    return results

