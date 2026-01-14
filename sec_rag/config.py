"""Configuration constants and paths."""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
INDEXES_DIR = DATA_DIR / "indexes"
GRAPHS_DIR = DATA_DIR / "graphs"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Ensure directories exist
for dir_path in [RAW_DIR, PARSED_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, INDEXES_DIR, GRAPHS_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# SEC API configuration
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip()
if not SEC_USER_AGENT:
    SEC_USER_AGENT = "Example Name example@example.com"

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

SEC_BASE_URL = "https://data.sec.gov"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

# Embedding configuration
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384  # all-MiniLM-L6-v2 dimension

# Chunking configuration
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "2200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "250"))
MIN_CHUNK_LENGTH = 200

# Retrieval configuration
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "6"))
MAX_RETRIEVAL_K = 50

# Ollama configuration
USE_OLLAMA = os.environ.get("USE_OLLAMA", "1").strip() not in ("0", "false", "False")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TIMEOUT = 300

# GraphRAG configuration
GRAPH_ALPHA = float(os.environ.get("GRAPH_ALPHA", "0.7"))  # Weight for cosine similarity vs graph boost
GRAPH_CO_OCCURRENCE_WINDOW = 1  # Entities in same chunk

# Risk themes taxonomy
RISK_THEMES = {
    "Supply Chain": "supply chain disruptions, logistics, manufacturing, suppliers, vendors, procurement",
    "Regulatory": "regulatory compliance, regulations, government oversight, SEC, FDA, regulatory changes",
    "Geopolitical": "geopolitical risks, international conflicts, trade wars, sanctions, political instability",
    "AI/Data Privacy": "artificial intelligence, machine learning, data privacy, GDPR, data protection, AI risks",
    "Cybersecurity": "cybersecurity, data breaches, cyber attacks, information security, hacking, ransomware",
    "Macro/Inflation": "inflation, economic conditions, macroeconomic factors, recession, interest rates",
    "Competition": "competitive risks, market competition, competitive pressures, market share",
    "Legal/IP": "legal proceedings, litigation, intellectual property, patents, trademarks, lawsuits",
}

# Default filing type
DEFAULT_FORM_TYPE = "10-K"

