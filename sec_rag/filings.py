"""SEC filing retrieval and metadata handling."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sec_rag.config import SEC_ARCHIVES_URL
from sec_rag.sec_client import get_company_submissions, sec_get
from sec_rag.utils import normalize_cik


@dataclass
class FilingMetadata:
    """Metadata for a SEC filing."""
    ticker: str
    cik: str
    form: str
    accession_number: str
    accession_nodash: str
    filing_date: str
    primary_document: str
    resolved_document: Optional[str] = None
    year: Optional[int] = None


def pick_recent_filing(
    submissions: Dict[str, Any],
    form_type: str = "10-K",
    year: Optional[int] = None
) -> Tuple[str, str, str]:
    """
    Pick the most recent filing of a given type, optionally filtered by year.
    
    Args:
        submissions: Company submissions JSON
        form_type: Form type (e.g., "10-K", "10-Q")
        year: Optional year filter (finds closest filing to that year)
        
    Returns:
        Tuple of (accession_nodash, primary_document, filing_date)
        
    Raises:
        ValueError: If no matching filing found
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    prims = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])
    
    candidates = []
    for f, a, p, d in zip(forms, accs, prims, dates):
        if f == form_type:
            filing_year = int(d.split("-")[0]) if d else None
            candidates.append((a.replace("-", ""), p, d, filing_year))
    
    if not candidates:
        raise ValueError(f"No recent {form_type} found in submissions JSON.")
    
    # If year specified, find closest match
    if year is not None:
        candidates.sort(key=lambda x: abs((x[3] or 0) - year))
        # Prefer exact match, then closest
        exact_matches = [c for c in candidates if c[3] == year]
        if exact_matches:
            candidates = exact_matches
    
    # Return most recent (first in list)
    accession_nodash, primary_doc, filing_date, _ = candidates[0]
    return accession_nodash, primary_doc, filing_date


def get_filing_index_json(cik: str, accession_nodash: str) -> Dict[str, Any]:
    """
    Fetch the filing index.json to resolve actual document names.
    
    Args:
        cik: Company CIK
        accession_nodash: Accession number without dashes
        
    Returns:
        Index JSON as dict
    """
    cik_int = str(int(cik))
    index_url = f"{SEC_ARCHIVES_URL}/{cik_int}/{accession_nodash}/index.json"
    return sec_get(index_url).json()


def pick_best_doc_from_index(index_json: Dict[str, Any], preferred: Optional[str] = None) -> str:
    """
    Pick the best document from filing index, preferring the primary document.
    
    Args:
        index_json: Filing index JSON
        preferred: Preferred document name (usually primary document)
        
    Returns:
        Best document filename
        
    Raises:
        ValueError: If no HTML documents found
    """
    items = index_json.get("directory", {}).get("item", [])
    names = [it.get("name") for it in items if it.get("name")]
    
    if preferred and preferred in names:
        return preferred
    
    htmls = [n for n in names if n.lower().endswith((".htm", ".html"))]
    if not htmls:
        raise ValueError("No .htm/.html documents found in filing index.json")
    
    # Sort by size (largest first)
    name_to_size = {it["name"]: int(it.get("size", 0) or 0) for it in items if "name" in it}
    htmls.sort(key=lambda n: name_to_size.get(n, 0), reverse=True)
    return htmls[0]


def get_filing_url(cik: str, accession_nodash: str, doc_name: str) -> str:
    """Construct the full URL for a filing document."""
    cik_int = str(int(cik))
    return f"{SEC_ARCHIVES_URL}/{cik_int}/{accession_nodash}/{doc_name}"


def fetch_filing_metadata(
    ticker: str,
    cik: str,
    form: str = "10-K",
    year: Optional[int] = None
) -> FilingMetadata:
    """
    Fetch metadata for a filing.
    
    Args:
        ticker: Company ticker symbol
        cik: Company CIK
        form: Form type (default: "10-K")
        year: Optional year filter
        
    Returns:
        FilingMetadata object
    """
    cik_normalized = normalize_cik(cik)
    submissions = get_company_submissions(cik_normalized)
    accession_nodash, primary_doc, filing_date = pick_recent_filing(submissions, form_type=form, year=year)
    
    # Resolve actual document name
    index_json = get_filing_index_json(cik_normalized, accession_nodash)
    resolved_doc = pick_best_doc_from_index(index_json, preferred=primary_doc)
    
    # Extract year from filing date
    filing_year = int(filing_date.split("-")[0]) if filing_date else None
    
    accession_number = "-".join([
        accession_nodash[:10],
        accession_nodash[10:12],
        accession_nodash[12:]
    ])
    
    return FilingMetadata(
        ticker=ticker,
        cik=cik_normalized,
        form=form,
        accession_number=accession_number,
        accession_nodash=accession_nodash,
        filing_date=filing_date,
        primary_document=primary_doc,
        resolved_document=resolved_doc,
        year=filing_year
    )

