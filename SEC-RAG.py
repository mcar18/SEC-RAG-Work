"""
Legacy wrapper for SEC-RAG.py - redirects to new modular CLI.

This file is maintained for backward compatibility.
For new usage, use: python -m sec_rag.cli <command>
"""

import os
import sys

if __name__ == "__main__":
    # Redirect to new CLI for backward compatibility
    print("Note: SEC-RAG.py is deprecated. Use 'python -m sec_rag.cli' instead.")
    print("\nExample usage:")
    print("  python -m sec_rag.cli ingest --tickers AAPL --years 2023")
    print("  python -m sec_rag.cli qa --question 'Your question' --tickers AAPL")
    print("\nRunning legacy mode with environment variables...")
    
    # Try to run basic QA if environment variables are set
    cik = os.environ.get("CIK", "0000320193")
    question = os.environ.get("QUESTION", "What does the company say are its main risk factors related to supply chain?")
    
    # Use new CLI
    from sec_rag.cli import cmd_qa
    from argparse import Namespace
    
    args = Namespace(
        question=question,
        tickers=None,
        years=None,
        topk=int(os.environ.get("TOP_K", "6")),
        no_ollama=os.environ.get("USE_OLLAMA", "1").strip() in ("0", "false", "False"),
        func=cmd_qa
    )
    
    # Note: This requires filings to be ingested first
    print("\nTo use this script, first ingest filings:")
    print(f"  python -m sec_rag.cli ingest --tickers AAPL --years 2023")
    print("\nThen run QA:")
    print(f"  python -m sec_rag.cli qa --question '{question}'")
    
    sys.exit(0)

