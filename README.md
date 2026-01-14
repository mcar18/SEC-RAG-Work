# SEC EDGAR RAG System

A Retrieval-Augmented Generation (RAG) system for querying SEC EDGAR filings using local embeddings and optional local LLM inference via Ollama. This project enables you to ask questions about company SEC filings (10-K, 10-Q) and get answers grounded in the actual filing documents.

## Overview

This system implements a complete RAG pipeline that:
1. **Fetches SEC filings** - Retrieves the most recent 10-K or 10-Q filing for a given company (by CIK)
2. **Resolves filing documents** - Uses the filing index.json to prevent 404 errors by finding the correct document
3. **Extracts and chunks text** - Converts HTML filings to clean text and splits them into overlapping chunks
4. **Creates embeddings** - Uses SentenceTransformers to generate vector embeddings for semantic search
5. **Retrieves relevant chunks** - Uses scikit-learn's NearestNeighbors for cosine similarity search (no FAISS required)
6. **Generates answers** - Optionally uses local Ollama LLM to generate grounded answers with citations

## Key Features

- **No API tokens required** - Uses local Ollama for LLM inference and local embeddings
- **No FAISS dependency** - Uses scikit-learn for vector search (suitable for up to tens of thousands of chunks)
- **Robust SEC API handling** - Includes retry logic and proper User-Agent headers
- **Citation support** - Answers include numbered citations [1], [2] referencing source chunks
- **Configurable via environment variables** - Easy customization without code changes

## Installation

### Prerequisites

- Python 3.8+
- Ollama installed and running (for answer generation)

### Python Dependencies

```bash
pip install -U requests beautifulsoup4 lxml tqdm numpy scikit-learn sentence-transformers
```

### Ollama Setup

1. Install [Ollama](https://ollama.ai/) for Windows
2. Pull a model (recommended: llama3.1:8b):
   ```bash
   ollama pull llama3.1:8b
   ```
3. Ensure Ollama server is running (usually automatic)

### SEC User Agent (Recommended)

The SEC recommends including a User-Agent header with your contact information. Set it via environment variable:

**PowerShell:**
```powershell
setx SEC_USER_AGENT "Your Name your.email@domain.com"
```
Restart your terminal after setting this.

## Usage

### Basic Usage

Run with default settings (Apple's 10-K):
```bash
python SEC-RAG.py
```

### Environment Variables

Customize behavior via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CIK` | `0000320193` | Company CIK (10-digit, zero-padded) |
| `FORM` | `10-K` | Filing type (`10-K` or `10-Q`) |
| `QUESTION` | Supply chain risk question | Your question about the filing |
| `TOP_K` | `6` | Number of chunks to retrieve |
| `USE_OLLAMA` | `1` | Set to `0` for retrieval-only mode |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model to use |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_CHARS` | `2200` | Characters per chunk |
| `CHUNK_OVERLAP` | `250` | Overlap between chunks |

### Example: Query Microsoft's 10-K

**PowerShell:**
```powershell
$env:CIK = "0000789019"
$env:FORM = "10-K"
$env:QUESTION = "What are Microsoft's main competitive risks?"
python SEC-RAG.py
```

**Command Prompt:**
```cmd
set CIK=0000789019
set FORM=10-K
set QUESTION=What are Microsoft's main competitive risks?
python SEC-RAG.py
```

### Example: Retrieval-Only Mode (No LLM)

```powershell
$env:USE_OLLAMA = "0"
python SEC-RAG.py
```

This will show the retrieved chunks and the prompt that would be sent to Ollama, but won't generate an answer.

## How It Works

### 1. Filing Retrieval
- Fetches company submissions JSON from `data.sec.gov`
- Finds the most recent filing of the specified type (10-K/10-Q)
- Extracts accession number, primary document, and filing date

### 2. Document Resolution
- Downloads the filing folder's `index.json` to find available documents
- Selects the best HTML document (prefers primary doc, falls back to largest HTML file)
- Prevents 404 errors by resolving the actual document path

### 3. Text Extraction
- Parses HTML using BeautifulSoup
- Removes scripts, styles, and noscript tags
- Cleans whitespace and normalizes text

### 4. Chunking
- Splits text into overlapping chunks (default: 2200 chars with 250 char overlap)
- Filters out chunks shorter than 200 characters

### 5. Embedding & Indexing
- Uses SentenceTransformers to create embeddings
- Normalizes embeddings for cosine similarity
- Builds a scikit-learn NearestNeighbors index

### 6. Retrieval
- Embeds the user's question
- Finds top-k most similar chunks using cosine similarity
- Returns chunks with similarity scores

### 7. Answer Generation (Optional)
- Builds a prompt with retrieved chunks and metadata
- Sends to local Ollama LLM
- Returns answer with citations referencing source chunks

## Architecture

```
┌─────────────────┐
│  SEC EDGAR API  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  Submissions    │────▶│  Filing HTML │
│  JSON           │     │  Resolution  │
└─────────────────┘     └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Text Extract│
                         │  & Chunking  │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Embeddings  │
                         │  (Sentence   │
                         │  Transformers)│
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Vector      │
                         │  Search      │
                         │  (sklearn)   │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Ollama LLM  │
                         │  (Optional)  │
                         └──────────────┘
```

## Code Structure

- **Configuration** (lines 54-74): Environment variables and defaults
- **Data Structures** (lines 80-83): `Chunk` dataclass for text and metadata
- **SEC HTTP Helpers** (lines 89-112): Polite GET requests with retry logic
- **Filing Resolution** (lines 132-175): Index.json parsing and HTML extraction
- **Chunking & Embeddings** (lines 181-238): Text chunking and vector indexing
- **Ollama Integration** (lines 269-293): Local LLM API calls
- **Main Pipeline** (lines 299-359): Orchestrates the entire RAG workflow

## Limitations

- **Vector Search**: Uses brute-force cosine similarity. Suitable for up to ~50K chunks. For larger datasets, consider FAISS or other approximate nearest neighbor libraries.
- **Chunking**: Simple character-based chunking. More sophisticated methods (sentence-aware, semantic chunking) could improve results.
- **Ollama Dependency**: Answer generation requires Ollama. Can run in retrieval-only mode without it.

## Troubleshooting

### Connection Errors to Ollama
- Ensure Ollama is installed and running
- Test with: `ollama run llama3.1:8b "hello"`
- Check `OLLAMA_URL` environment variable

### SEC API Rate Limiting
- Set a proper `SEC_USER_AGENT` with your contact info
- The script includes automatic retry logic for transient errors

### Empty Chunks
- Check that the filing HTML contains extractable text
- Verify the resolved document is the correct one

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues or pull requests for improvements!

