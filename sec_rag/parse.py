"""HTML parsing and text extraction from SEC filings."""

import re
import warnings
from pathlib import Path
from typing import Optional, Tuple

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Suppress XML parsed as HTML warnings (SEC filings are often XHTML)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from sec_rag.config import PARSED_DIR
from sec_rag.filings import FilingMetadata, get_filing_url
from sec_rag.sec_client import sec_get
from sec_rag.utils import load_json, save_json


def html_to_text(html: str) -> str:
    """
    Extract clean text from HTML.
    
    Args:
        html: HTML content
        
    Returns:
        Clean text
    """
    soup = BeautifulSoup(html, "lxml")
    
    # Remove scripts, styles, and noscript tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    
    # Extract text
    text = soup.get_text("\n")
    
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    
    return text.strip()


def download_and_parse_filing(metadata: FilingMetadata, use_cache: bool = True) -> Tuple[str, str]:
    """
    Download and parse a filing, with optional caching.
    
    Args:
        metadata: FilingMetadata object
        use_cache: Whether to use cached parsed text if available
        
    Returns:
        Tuple of (parsed_text, resolved_document_name)
    """
    # Check cache
    if use_cache:
        cache_path = PARSED_DIR / f"{metadata.ticker}_{metadata.cik}_{metadata.form}_{metadata.year}_{metadata.accession_nodash}.txt"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read(), metadata.resolved_document or metadata.primary_document
    
    # Download HTML
    if not metadata.resolved_document:
        from sec_rag.filings import get_filing_index_json, pick_best_doc_from_index
        index_json = get_filing_index_json(metadata.cik, metadata.accession_nodash)
        metadata.resolved_document = pick_best_doc_from_index(index_json, preferred=metadata.primary_document)
    
    url = get_filing_url(metadata.cik, metadata.accession_nodash, metadata.resolved_document)
    response = sec_get(url)
    html = response.text
    
    # Parse to text
    text = html_to_text(html)
    
    # Cache parsed text
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
    
    return text, metadata.resolved_document


def load_parsed_filing(metadata: FilingMetadata) -> Optional[str]:
    """Load cached parsed filing text."""
    cache_path = PARSED_DIR / f"{metadata.ticker}_{metadata.cik}_{metadata.form}_{metadata.year}_{metadata.accession_nodash}.txt"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

