"""Command-line interface for SEC RAG."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from sec_rag.config import DEFAULT_TOP_K, OUTPUTS_DIR
from sec_rag.embed import Embedder
from sec_rag.index import SklearnVectorIndex
from sec_rag.ingest import ingest_multiple, ticker_to_cik
from sec_rag.rag import answer_question
from sec_rag.sec_client import check_ollama_available


def cmd_ingest(args):
    """Ingest filings for given tickers and years."""
    tickers = args.tickers
    years = list(range(args.years[0], args.years[1] + 1)) if len(args.years) == 2 else args.years
    
    print(f"Ingesting filings for tickers: {tickers}")
    print(f"Years: {years}")
    print(f"Form: {args.form}")
    
    # Build CIK map if provided
    cik_map = {}
    if args.ciks:
        if len(args.ciks) != len(tickers):
            print("Error: Number of CIKs must match number of tickers")
            sys.exit(1)
        cik_map = dict(zip(tickers, args.ciks))
    
    results = ingest_multiple(
        tickers=tickers,
        years=years,
        form=args.form,
        cik_map=cik_map,
        use_cache=True
    )
    
    print(f"\n{'='*60}")
    print("Ingestion complete!")
    print(f"{'='*60}")
    for ticker, year_data in results.items():
        print(f"{ticker}: {len(year_data)} filings ingested")


def cmd_qa(args):
    """Answer a question using RAG."""
    # Check Ollama
    if not check_ollama_available():
        print("Warning: Ollama not available. Running in retrieval-only mode.")
        use_ollama = False
    else:
        use_ollama = not args.no_ollama
    
    # Build filter
    filter_meta = {}
    if args.tickers and len(args.tickers) == 1:
        filter_meta["ticker"] = args.tickers[0]
    if args.years and len(args.years) == 1:
        filter_meta["year"] = args.years[0]
    
    # Load or build index
    print("Loading aggregated index...")
    from sec_rag.index_manager import IndexManager
    
    manager = IndexManager()
    
    # Determine years from args or use all available
    if args.years:
        if len(args.years) == 2:
            years = list(range(args.years[0], args.years[1] + 1))
        else:
            years = args.years
    else:
        # Try to find available years from disk
        from sec_rag.config import INDEXES_DIR
        years = []
        for path in INDEXES_DIR.glob("*.pkl"):
            parts = path.stem.split("_")
            if len(parts) >= 4:
                try:
                    years.append(int(parts[3]))
                except ValueError:
                    pass
        years = sorted(set(years)) if years else [2020, 2021, 2022, 2023, 2024]
    
    # Determine tickers
    if args.tickers:
        tickers = args.tickers
    else:
        # Try to find available tickers from disk
        from sec_rag.config import INDEXES_DIR
        tickers = []
        for path in INDEXES_DIR.glob("*.pkl"):
            parts = path.stem.split("_")
            if parts:
                tickers.append(parts[0])
        tickers = sorted(set(tickers)) if tickers else []
    
    if not tickers:
        print("Error: No filings found. Please run 'ingest' command first.")
        sys.exit(1)
    
    print(f"Loading indexes for tickers: {tickers}, years: {years}")
    index = manager.aggregate_indexes(tickers, years)
    
    if len(index.chunks) == 0:
        print("Error: No chunks found in index. Please run 'ingest' command first.")
        sys.exit(1)
    
    print(f"Index loaded with {len(index.chunks)} chunks")
    print(f"\nQuestion: {args.question}")
    print(f"Top-K: {args.topk}")
    if filter_meta:
        print(f"Filters: {filter_meta}")
    
    # Answer question
    answer, retrieved = answer_question(
        args.question,
        index,
        top_k=args.topk,
        filter_meta=filter_meta if filter_meta else None,
        use_ollama=use_ollama
    )
    
    print(f"\n{'='*60}")
    print("Answer:")
    print(f"{'='*60}")
    print(answer)
    
    if retrieved:
        print(f"\n{'='*60}")
        print("Retrieved Chunks:")
        print(f"{'='*60}")
        for i, (chunk, score) in enumerate(retrieved[:3], 1):
            print(f"\n[{i}] Score: {score:.3f}")
            print(f"Ticker: {chunk.meta.get('ticker')}, Year: {chunk.meta.get('year')}")
            print(f"Preview: {chunk.text[:200]}...")


def cmd_analytics(args):
    """Generate risk theme analytics."""
    from sec_rag.analytics import generate_analytics
    from sec_rag.index_manager import IndexManager
    from sec_rag.ingest import ingest_multiple, ticker_to_cik
    
    tickers = args.tickers
    years = list(range(args.years[0], args.years[1] + 1)) if len(args.years) == 2 else args.years
    
    print(f"Generating analytics for tickers: {tickers}, years: {years}")
    
    # Load filings data
    filings_data = {}
    manager = IndexManager()
    
    for ticker in tickers:
        cik = ticker_to_cik(ticker)
        if not cik:
            print(f"Warning: Could not resolve CIK for {ticker}, skipping...")
            continue
        
        filings_data[ticker] = {}
        
        for year in years:
            # Try to load from disk
            from sec_rag.config import INDEXES_DIR
            index_files = list(INDEXES_DIR.glob(f"{ticker}_{cik}_10-K_{year}_*.pkl"))
            if not index_files:
                print(f"Warning: No filing found for {ticker} {year}. Run 'ingest' first.")
                continue
            
            # Load chunks and metadata from disk
            from sec_rag.config import CHUNKS_DIR
            from sec_rag.chunking import load_chunks
            from sec_rag.filings import FilingMetadata
            
            chunk_files = list(CHUNKS_DIR.glob(f"{ticker}_{cik}_10-K_{year}_*.json"))
            if chunk_files:
                chunks = load_chunks(chunk_files[0])
                index = manager.load_filing_index(ticker, cik, "10-K", year, chunks[0].meta.get("accession_nodash", ""))
                
                if index:
                    # Create metadata from chunk
                    meta_dict = chunks[0].meta
                    metadata = FilingMetadata(
                        ticker=meta_dict.get("ticker", ticker),
                        cik=meta_dict.get("cik", cik),
                        form=meta_dict.get("form", "10-K"),
                        accession_number=meta_dict.get("accession_number", ""),
                        accession_nodash=meta_dict.get("accession_nodash", ""),
                        filing_date=meta_dict.get("filing_date", ""),
                        primary_document=meta_dict.get("primary_document", ""),
                        resolved_document=meta_dict.get("resolved_document"),
                        year=year
                    )
                    filings_data[ticker][year] = (metadata, chunks, index)
    
    if not filings_data:
        print("Error: No filings data found. Please run 'ingest' command first.")
        sys.exit(1)
    
    # Generate analytics
    df = generate_analytics(filings_data, tickers, years)
    print(f"\nAnalytics complete! Results saved to {OUTPUTS_DIR}")


def cmd_summarize(args):
    """Generate comparative summaries."""
    from sec_rag.summarize import generate_summaries
    from sec_rag.index_manager import IndexManager
    from sec_rag.ingest import ticker_to_cik
    
    tickers = args.tickers
    years = list(range(args.years[0], args.years[1] + 1)) if len(args.years) == 2 else args.years
    
    print(f"Generating summaries for tickers: {tickers}, years: {years}")
    
    # Load filings data (similar to analytics)
    filings_data = {}
    
    for ticker in tickers:
        cik = ticker_to_cik(ticker)
        if not cik:
            print(f"Warning: Could not resolve CIK for {ticker}, skipping...")
            continue
        
        filings_data[ticker] = {}
        
        for year in years:
            from sec_rag.config import CHUNKS_DIR, INDEXES_DIR
            from sec_rag.chunking import load_chunks
            from sec_rag.filings import FilingMetadata
            from sec_rag.index_manager import IndexManager
            
            manager = IndexManager()
            chunk_files = list(CHUNKS_DIR.glob(f"{ticker}_{cik}_10-K_{year}_*.json"))
            
            if chunk_files:
                chunks = load_chunks(chunk_files[0])
                accession_nodash = chunks[0].meta.get("accession_nodash", "")
                index = manager.load_filing_index(ticker, cik, "10-K", year, accession_nodash)
                
                if index and chunks:
                    meta_dict = chunks[0].meta
                    metadata = FilingMetadata(
                        ticker=meta_dict.get("ticker", ticker),
                        cik=meta_dict.get("cik", cik),
                        form=meta_dict.get("form", "10-K"),
                        accession_number=meta_dict.get("accession_number", ""),
                        accession_nodash=accession_nodash,
                        filing_date=meta_dict.get("filing_date", ""),
                        primary_document=meta_dict.get("primary_document", ""),
                        resolved_document=meta_dict.get("resolved_document"),
                        year=year
                    )
                    filings_data[ticker][year] = (metadata, chunks, index)
    
    if not filings_data:
        print("Error: No filings data found. Please run 'ingest' command first.")
        sys.exit(1)
    
    generate_summaries(filings_data)
    print(f"\nSummaries complete! Results saved to {OUTPUTS_DIR / 'summaries'}")


def cmd_graph(args):
    """Build entity graph for GraphRAG."""
    from sec_rag.graph import build_graphs_for_filings
    from sec_rag.ingest import ticker_to_cik
    
    tickers = args.tickers
    years = list(range(args.years[0], args.years[1] + 1)) if len(args.years) == 2 else args.years
    
    print(f"Building graphs for tickers: {tickers}, years: {years}")
    
    # Load filings data
    filings_data = {}
    
    for ticker in tickers:
        cik = ticker_to_cik(ticker)
        if not cik:
            print(f"Warning: Could not resolve CIK for {ticker}, skipping...")
            continue
        
        filings_data[ticker] = {}
        
        for year in years:
            from sec_rag.config import CHUNKS_DIR, INDEXES_DIR
            from sec_rag.chunking import load_chunks
            from sec_rag.filings import FilingMetadata
            from sec_rag.index_manager import IndexManager
            
            manager = IndexManager()
            chunk_files = list(CHUNKS_DIR.glob(f"{ticker}_{cik}_10-K_{year}_*.json"))
            
            if chunk_files:
                chunks = load_chunks(chunk_files[0])
                accession_nodash = chunks[0].meta.get("accession_nodash", "")
                index = manager.load_filing_index(ticker, cik, "10-K", year, accession_nodash)
                
                if index and chunks:
                    meta_dict = chunks[0].meta
                    metadata = FilingMetadata(
                        ticker=meta_dict.get("ticker", ticker),
                        cik=meta_dict.get("cik", cik),
                        form=meta_dict.get("form", "10-K"),
                        accession_number=meta_dict.get("accession_number", ""),
                        accession_nodash=accession_nodash,
                        filing_date=meta_dict.get("filing_date", ""),
                        primary_document=meta_dict.get("primary_document", ""),
                        resolved_document=meta_dict.get("resolved_document"),
                        year=year
                    )
                    filings_data[ticker][year] = (metadata, chunks, index)
    
    if not filings_data:
        print("Error: No filings data found. Please run 'ingest' command first.")
        sys.exit(1)
    
    graphs = build_graphs_for_filings(filings_data)
    print(f"\nGraphs built! {len(graphs)} graphs created.")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="SEC RAG - Query SEC filings with RAG")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest SEC filings")
    ingest_parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols")
    ingest_parser.add_argument("--years", type=int, nargs="+", required=True, help="Years (single or range)")
    ingest_parser.add_argument("--form", default="10-K", help="Form type (default: 10-K)")
    ingest_parser.add_argument("--ciks", nargs="+", help="Optional CIKs matching tickers")
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # QA command
    qa_parser = subparsers.add_parser("qa", help="Answer a question")
    qa_parser.add_argument("--question", required=True, help="Question to answer")
    qa_parser.add_argument("--tickers", nargs="+", help="Filter by ticker(s)")
    qa_parser.add_argument("--years", type=int, nargs="+", help="Filter by year(s)")
    qa_parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K, help="Top-K chunks (default: 6)")
    qa_parser.add_argument("--no-ollama", action="store_true", help="Disable Ollama (retrieval only)")
    qa_parser.set_defaults(func=cmd_qa, use_ollama=True)
    
    # Analytics command
    analytics_parser = subparsers.add_parser("analytics", help="Generate risk theme analytics")
    analytics_parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols")
    analytics_parser.add_argument("--years", type=int, nargs="+", required=True, help="Years")
    analytics_parser.set_defaults(func=cmd_analytics)
    
    # Summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Generate comparative summaries")
    summarize_parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols")
    summarize_parser.add_argument("--years", type=int, nargs="+", required=True, help="Years")
    summarize_parser.set_defaults(func=cmd_summarize)
    
    # Graph command
    graph_parser = subparsers.add_parser("graph", help="Build entity graph")
    graph_parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols")
    graph_parser.add_argument("--years", type=int, nargs="+", required=True, help="Years")
    graph_parser.set_defaults(func=cmd_graph)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()

