"""SEC EDGAR API client with retry logic and rate limiting."""

import time
from typing import Any, Dict, Optional

import requests

from sec_rag.config import SEC_HEADERS


def sec_get(url: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 6) -> requests.Response:
    """
    Polite GET with retries/backoff for transient SEC errors.
    
    Args:
        url: URL to fetch
        params: Optional query parameters
        max_retries: Maximum number of retry attempts
        
    Returns:
        Response object
        
    Raises:
        requests.HTTPError: If request fails after all retries
    """
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=SEC_HEADERS, params=params, timeout=45)
            
            if r.status_code == 200:
                return r
            
            # Retry on transient errors
            if r.status_code in (403, 429, 500, 502, 503, 504):
                if attempt < max_retries - 1:
                    time.sleep(1.2 * (attempt + 1))
                    continue
            
            # For other errors, raise immediately
            r.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1.2 * (attempt + 1))
                continue
            raise
    
    r.raise_for_status()
    return r


def get_company_submissions(cik: str) -> Dict[str, Any]:
    """
    Fetch company submissions JSON from SEC.
    
    Args:
        cik: Company CIK (10-digit, zero-padded)
        
    Returns:
        Submissions JSON as dict
    """
    from sec_rag.config import SEC_BASE_URL
    
    cik10 = cik.zfill(10)
    url = f"{SEC_BASE_URL}/submissions/CIK{cik10}.json"
    return sec_get(url).json()


def check_ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    from sec_rag.config import OLLAMA_URL
    
    try:
        # Try to get available models
        base_url = OLLAMA_URL.replace("/api/generate", "")
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

