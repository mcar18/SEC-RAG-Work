# SEC RAG - Portfolio-Grade SEC Filing Analysis System

A comprehensive Retrieval-Augmented Generation (RAG) system for analyzing SEC EDGAR filings with multi-ticker, multi-year support, risk theme analytics, comparative summaries, and GraphRAG capabilities.

## Features

- **Multi-Ticker & Multi-Year Support**: Ingest and analyze filings across multiple companies and years
- **RAG QA**: Ask questions about filings with grounded answers and citations
- **Risk Theme Scoring**: Automated scoring of 8 risk themes (Supply Chain, Regulatory, Geopolitical, AI/Data Privacy, Cybersecurity, Macro/Inflation, Competition, Legal/IP)
- **Analytics & Visualizations**: Generate charts and heatmaps for risk analysis
- **Comparative Summaries**: Year-over-year comparisons and trend analysis
- **GraphRAG**: Entity graph construction with graph-aware retrieval
- **Local-Only**: No API keys required - uses local Ollama and embeddings
- **Caching**: Persistent caching for fast repeated runs

## Installation

### Prerequisites

- Python 3.13+ (Windows-friendly)
- Ollama installed and running

### Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and configure Ollama:**
   ```bash
   # Download from https://ollama.ai/
   ollama pull llama3.1:8b
   ```

3. **Set SEC User-Agent (recommended):**
   ```powershell
   setx SEC_USER_AGENT "Your Name your.email@domain.com"
   ```
   Restart terminal after setting.

## Quick Start

### 1. Ingest Filings

Download and process SEC filings for multiple companies and years:

```bash
python -m sec_rag.cli ingest --tickers AAPL MSFT --years 2020 2024
```

This will:
- Download 10-K filings for Apple and Microsoft from 2020-2024
- Parse HTML to text
- Chunk and embed documents
- Build vector indexes
- Cache everything to disk

### 2. Ask Questions

Query the ingested filings:

```bash
python -m sec_rag.cli qa --question "What are the main supply chain risks?" --tickers AAPL --topk 6
```

Filter by year:
```bash
python -m sec_rag.cli qa --question "What changed in risk factors?" --tickers AAPL --years 2023
```

### 3. Generate Analytics

Score risk themes and generate visualizations:

```bash
python -m sec_rag.cli analytics --tickers AAPL MSFT --years 2020 2024
```

Outputs:
- `data/outputs/theme_scores.csv` - Theme scores for all filings
- `data/outputs/timeline_*.png` - Timeline charts per theme
- `data/outputs/heatmap_*.png` - Company x Theme heatmaps
- `data/outputs/top_movers.png` - Year-over-year changes

### 4. Generate Summaries

Create comparative summaries:

```bash
python -m sec_rag.cli summarize --tickers AAPL --years 2020 2024
```

Outputs markdown summaries in `data/outputs/summaries/` with:
- Top 5 risk factors
- Changes vs prior year
- Notable new/accelerating risks

### 5. Build Entity Graphs

Construct entity graphs for GraphRAG:

```bash
python -m sec_rag.cli graph --tickers AAPL MSFT --years 2020 2024
```

Graphs saved to `data/graphs/` for graph-aware retrieval.

## Project Structure

```
SEC-RAG-Work/
├── sec_rag/              # Main package
│   ├── __init__.py
│   ├── config.py         # Configuration constants
│   ├── sec_client.py      # SEC API client
│   ├── filings.py         # Filing metadata handling
│   ├── parse.py           # HTML parsing
│   ├── chunking.py        # Text chunking
│   ├── embed.py           # Embedding generation
│   ├── index.py           # Vector index (sklearn)
│   ├── index_manager.py   # Multi-filing index aggregation
│   ├── rag.py             # RAG pipeline
│   ├── ollama_client.py   # Ollama LLM client
│   ├── ingest.py          # Ingestion pipeline
│   ├── themes.py          # Risk theme scoring
│   ├── analytics.py       # Analytics & visualizations
│   ├── summarize.py       # Comparative summaries
│   ├── graph.py           # GraphRAG implementation
│   ├── utils.py           # Utility functions
│   └── cli.py             # Command-line interface
├── data/                  # Data directories
│   ├── raw/              # Raw HTML filings
│   ├── parsed/           # Parsed text
│   ├── chunks/           # Text chunks
│   ├── embeddings/       # Embedding vectors
│   ├── indexes/          # Vector indexes
│   ├── graphs/           # Entity graphs
│   └── outputs/          # Results (CSV, charts, summaries)
├── requirements.txt
├── .gitignore
└── README.md
```

## CLI Commands

### `ingest`

Ingest SEC filings for processing.

```bash
python -m sec_rag.cli ingest --tickers TICKER1 TICKER2 --years YEAR1 YEAR2 [--form FORM] [--ciks CIK1 CIK2]
```

**Arguments:**
- `--tickers`: Ticker symbols (space-separated)
- `--years`: Years (single year or range like `2020 2024`)
- `--form`: Form type (default: `10-K`)
- `--ciks`: Optional CIKs matching tickers

**Example:**
```bash
python -m sec_rag.cli ingest --tickers AAPL MSFT GOOGL --years 2022 2024
```

### `qa`

Answer questions using RAG.

```bash
python -m sec_rag.cli qa --question "QUESTION" [--tickers TICKER] [--years YEAR] [--topk K] [--no-ollama]
```

**Arguments:**
- `--question`: Question to answer (required)
- `--tickers`: Filter by ticker(s)
- `--years`: Filter by year(s)
- `--topk`: Number of chunks to retrieve (default: 6)
- `--no-ollama`: Disable Ollama (retrieval-only mode)

**Example:**
```bash
python -m sec_rag.cli qa --question "What are the cybersecurity risks?" --tickers AAPL --years 2023
```

### `analytics`

Generate risk theme analytics and visualizations.

```bash
python -m sec_rag.cli analytics --tickers TICKER1 TICKER2 --years YEAR1 YEAR2
```

**Outputs:**
- CSV file with theme scores
- Timeline charts per theme
- Heatmaps by company and year
- Top movers chart

### `summarize`

Generate comparative summaries across years.

```bash
python -m sec_rag.cli summarize --tickers TICKER1 TICKER2 --years YEAR1 YEAR2
```

**Outputs:** Markdown summaries in `data/outputs/summaries/`

### `graph`

Build entity graphs for GraphRAG.

```bash
python -m sec_rag.cli graph --tickers TICKER1 TICKER2 --years YEAR1 YEAR2
```

**Outputs:** Entity graphs in `data/graphs/`

## Risk Themes

The system scores 8 predefined risk themes:

1. **Supply Chain** - Supply chain disruptions, logistics, manufacturing
2. **Regulatory** - Regulatory compliance, government oversight
3. **Geopolitical** - International conflicts, trade wars, sanctions
4. **AI/Data Privacy** - AI risks, data privacy, GDPR
5. **Cybersecurity** - Data breaches, cyber attacks, information security
6. **Macro/Inflation** - Economic conditions, inflation, recession
7. **Competition** - Market competition, competitive pressures
8. **Legal/IP** - Litigation, intellectual property, lawsuits

## Configuration

Edit `sec_rag/config.py` or set environment variables:

- `SEC_USER_AGENT` - SEC API user agent
- `EMBED_MODEL` - SentenceTransformer model (default: `all-MiniLM-L6-v2`)
- `CHUNK_CHARS` - Characters per chunk (default: 2200)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 250)
- `OLLAMA_MODEL` - Ollama model (default: `llama3.1:8b`)
- `GRAPH_ALPHA` - GraphRAG weight (default: 0.7)

## Caching

All intermediate results are cached to disk:
- Raw HTML → `data/raw/`
- Parsed text → `data/parsed/`
- Chunks → `data/chunks/`
- Embeddings → `data/embeddings/`
- Indexes → `data/indexes/`
- Graphs → `data/graphs/`

Re-running commands uses cached data for fast execution.

## GraphRAG

The system includes a lightweight GraphRAG implementation:

- **Entity Extraction**: Extracts countries, organizations, products/tech, regulations
- **Graph Construction**: Builds co-occurrence graphs from chunks
- **Graph-Aware Retrieval**: Boosts chunks connected to query entities
- **Scoring**: `final_score = alpha * cosine_sim + (1-alpha) * graph_boost`

## Limitations

- **Vector Search**: Uses brute-force cosine similarity (suitable for ~50K chunks)
- **Entity Extraction**: Simple regex-based (can be enhanced with spaCy)
- **Ticker Mapping**: Includes common tickers; add more in `sec_rag/ingest.py`

## Troubleshooting

### Ollama Connection Error
- Ensure Ollama is running: `ollama run llama3.1:8b "hello"`
- Check `OLLAMA_URL` in config

### SEC API Rate Limiting
- Set proper `SEC_USER_AGENT` with your contact info
- Script includes automatic retry logic

### No Filings Found
- Verify ticker → CIK mapping in `sec_rag/ingest.py`
- Provide CIKs directly via `--ciks` argument

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues or pull requests for improvements!
