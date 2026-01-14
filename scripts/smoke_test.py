"""Smoke test to verify environment and basic functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from sec_rag import config
        from sec_rag import sec_client
        from sec_rag import filings
        from sec_rag import parse
        from sec_rag import chunking
        from sec_rag import embed
        from sec_rag import index
        from sec_rag import rag
        from sec_rag import ollama_client
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_ollama():
    """Test Ollama connection."""
    print("\nTesting Ollama connection...")
    try:
        from sec_rag.sec_client import check_ollama_available
        if check_ollama_available():
            print("✓ Ollama is reachable")
            return True
        else:
            print("⚠ Ollama not reachable (this is OK if you're running retrieval-only)")
            return False
    except Exception as e:
        print(f"⚠ Ollama check failed: {e}")
        return False

def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    try:
        from sec_rag import config
        print(f"✓ Data directory: {config.DATA_DIR}")
        print(f"✓ Embed model: {config.EMBED_MODEL_NAME}")
        print(f"✓ Ollama model: {config.OLLAMA_MODEL}")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SEC RAG Smoke Test")
    print("=" * 60)
    
    results = []
    results.append(test_imports())
    results.append(test_config())
    results.append(test_ollama())
    
    print("\n" + "=" * 60)
    if all(results[:2]):  # Imports and config are required
        print("✓ Basic setup OK")
    else:
        print("✗ Setup issues detected")
        sys.exit(1)
    print("=" * 60)

