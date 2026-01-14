"""Comparative summaries across years and companies."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sec_rag.chunking import Chunk
from sec_rag.config import OUTPUTS_DIR
from sec_rag.index import SklearnVectorIndex
from sec_rag.ollama_client import call_ollama
from sec_rag.rag import build_prompt


def generate_filing_summary(
    metadata,
    chunks: List[Chunk],
    index: SklearnVectorIndex,
    prior_chunks: Optional[List[Chunk]] = None
) -> str:
    """
    Generate a structured summary for a filing.
    
    Args:
        metadata: FilingMetadata object
        chunks: Chunks from current filing
        index: Vector index (can be aggregated)
        prior_chunks: Optional chunks from prior year for comparison
        
    Returns:
        Markdown summary
    """
    # Top 5 risks
    risks_query = "What are the main risk factors and concerns mentioned in this filing?"
    risks_retrieved = index.search(risks_query, top_k=10, filter_meta={
        "ticker": metadata.ticker,
        "year": metadata.year
    })
    
    risks_prompt = build_prompt(
        "List the top 5 most significant risk factors mentioned. Be specific and cite sources.",
        risks_retrieved[:5]
    )
    
    risks_summary = call_ollama(risks_prompt)
    
    # What changed vs prior year
    changes_summary = ""
    if prior_chunks and len(prior_chunks) > 0:
        # Build a comparison prompt
        current_context = "\n\n".join([c.text[:500] for c in chunks[:5]])
        prior_context = "\n\n".join([c.text[:500] for c in prior_chunks[:5]])
        
        changes_prompt = f"""Compare the risk factors and key concerns between two SEC filings.

Current Filing ({metadata.year}):
{current_context}

Prior Filing:
{prior_context}

Identify:
1. What risks are new or have increased?
2. What risks have decreased or been removed?
3. Notable changes in language or emphasis?

Be specific and cite examples."""
        
        changes_summary = call_ollama(changes_prompt)
    
    # Notable new/accelerating risks
    accelerating_query = "What risks are new, accelerating, or have significantly increased in severity?"
    accelerating_retrieved = index.search(accelerating_query, top_k=8, filter_meta={
        "ticker": metadata.ticker,
        "year": metadata.year
    })
    
    accelerating_prompt = build_prompt(
        "Identify and describe any new, accelerating, or significantly worsening risk factors. Explain why they are notable.",
        accelerating_retrieved[:5]
    )
    
    accelerating_summary = call_ollama(accelerating_prompt)
    
    # Compile markdown
    markdown = f"""# SEC Filing Summary: {metadata.ticker} {metadata.form} {metadata.year}

**Filing Date:** {metadata.filing_date}  
**Accession Number:** {metadata.accession_number}  
**Document:** {metadata.resolved_document}

---

## Top 5 Risk Factors

{risks_summary}

---

## Changes vs Prior Year

{changes_summary if changes_summary else "*No prior year data available for comparison*"}

---

## Notable New/Accelerating Risks

{accelerating_summary}

---

*Generated using SEC RAG system*
"""
    
    return markdown


def generate_summaries(
    filings_data: Dict[str, Dict[int, Tuple]],
    output_dir: Optional[Path] = None
) -> None:
    """
    Generate summaries for all filings.
    
    Args:
        filings_data: Nested dict {ticker: {year: (metadata, chunks, index)}}
        output_dir: Optional output directory
    """
    if output_dir is None:
        output_dir = OUTPUTS_DIR / "summaries"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build aggregated index for better retrieval
    from sec_rag.index_manager import IndexManager
    manager = IndexManager()
    
    all_tickers = list(filings_data.keys())
    all_years = []
    for year_data in filings_data.values():
        all_years.extend(year_data.keys())
    all_years = sorted(set(all_years))
    
    print("Building aggregated index for summaries...")
    aggregated_index = manager.aggregate_indexes(all_tickers, all_years)
    
    for ticker, year_data in filings_data.items():
        sorted_years = sorted(year_data.keys())
        
        for year in sorted_years:
            metadata, chunks, index = year_data[year]
            
            # Get prior year chunks if available
            prior_chunks = None
            if year > min(sorted_years):
                prev_year = sorted_years[sorted_years.index(year) - 1]
                if prev_year in year_data:
                    _, prior_chunks, _ = year_data[prev_year]
            
            print(f"Generating summary for {ticker} {year}...")
            
            summary = generate_filing_summary(
                metadata,
                chunks,
                aggregated_index,
                prior_chunks=prior_chunks
            )
            
            # Save summary
            summary_path = output_dir / f"{ticker}_{year}_summary.md"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            
            print(f"Saved summary to {summary_path}")

