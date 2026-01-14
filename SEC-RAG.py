"""
SEC EDGAR RAG (No FAISS) + Local Ollama (No API tokens)

What this script does:
1) Pulls a company's most recent 10-K or 10-Q metadata from SEC submissions JSON
2) Resolves the actual filing HTML via filing folder index.json (prevents 404s)
3) Extracts text from HTML
4) Chunks the text
5) Embeds chunks with SentenceTransformers
6) Retrieves top-k chunks with scikit-learn NearestNeighbors (cosine)
7) Generates an answer via LOCAL Ollama (optional), with citations [1], [2], ...

Install:
  pip install -U requests beautifulsoup4 lxml tqdm numpy scikit-learn sentence-transformers

SEC_USER_AGENT (recommended):
  PowerShell:
    setx SEC_USER_AGENT "Your Name your.email@domain.com"
  Restart terminal.

Ollama:
  - Install Ollama for Windows
  - ollama pull llama3.1:8b
  - Ensure Ollama server is running (usually automatic)

Run:
  python sec_rag_ollama.py

Env overrides:
  setx CIK "0000320193"
  setx FORM "10-K"
  setx QUESTION "What does the company say are its main supply chain risks?"
  setx TOP_K "6"
  setx USE_OLLAMA "1"           (0 = retrieval-only)
  setx OLLAMA_MODEL "llama3.1:8b"
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip()
if not SEC_USER_AGENT:
    # You should set a real one; SEC may throttle without it.
    SEC_USER_AGENT = "Example Name example@example.com"

HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("TOP_K", "6"))

USE_OLLAMA = os.environ.get("USE_OLLAMA", "1").strip() not in ("0", "false", "False")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "2200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "250"))


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Chunk:
    text: str
    meta: Dict[str, Any]


# -----------------------------
# SEC HTTP helpers
# -----------------------------
def sec_get(url: str, params: Dict[str, Any] | None = None) -> requests.Response:
    """
    Polite GET with retries/backoff for transient SEC errors.
    """
    for attempt in range(6):
        r = requests.get(url, headers=HEADERS, params=params, timeout=45)

        if r.status_code == 200:
            return r

        if r.status_code in (403, 429, 500, 502, 503, 504):
            time.sleep(1.2 * (attempt + 1))
            continue

        r.raise_for_status()

    r.raise_for_status()
    return r


def get_company_submissions(cik: str) -> Dict[str, Any]:
    cik10 = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    return sec_get(url).json()


def pick_recent_filing(submissions: Dict[str, Any], form_type: str = "10-K") -> Tuple[str, str, str]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    prims = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])

    for f, a, p, d in zip(forms, accs, prims, dates):
        if f == form_type:
            return a.replace("-", ""), p, d

    raise ValueError(f"No recent {form_type} found in submissions JSON.")


# -----------------------------
# Filing resolution (prevents 404)
# -----------------------------
def get_filing_index_json(cik: str, accession_nodash: str) -> dict:
    cik_int = str(int(cik))
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/index.json"
    return sec_get(index_url).json()


def pick_best_doc_from_index(index_json: dict, preferred: str | None = None) -> str:
    items = index_json.get("directory", {}).get("item", [])
    names = [it.get("name") for it in items if it.get("name")]

    if preferred and preferred in names:
        return preferred

    htmls = [n for n in names if n.lower().endswith((".htm", ".html"))]
    if not htmls:
        raise ValueError("No .htm/.html documents found in filing index.json")

    name_to_size = {it["name"]: int(it.get("size", 0) or 0) for it in items if "name" in it}
    htmls.sort(key=lambda n: name_to_size.get(n, 0), reverse=True)
    return htmls[0]


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # If it's XHTML/XML-ish, this warning is harmless for MVP.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def download_filing_text_resolved(cik: str, accession_nodash: str, primary_doc: str) -> Tuple[str, str]:
    idx = get_filing_index_json(cik, accession_nodash)
    resolved_doc = pick_best_doc_from_index(idx, preferred=primary_doc)

    cik_int = str(int(cik))
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{resolved_doc}"
    r = sec_get(url)

    return html_to_text(r.text), resolved_doc


# -----------------------------
# Chunking + Embeddings + Retrieval
# -----------------------------
def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


class SklearnVectorIndex:
    """
    Cosine similarity search using scikit-learn brute force.
    Great for MVPs with up to tens of thousands of chunks.
    """

    def __init__(self, embed_model_name: str = EMBED_MODEL_NAME):
        self.embedder = SentenceTransformer(embed_model_name)
        self.embeddings: np.ndarray | None = None
        self.nn: NearestNeighbors | None = None
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk]) -> None:
        texts = [c.text for c in chunks]
        embs = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        embs = np.asarray(embs, dtype=np.float32)

        self.embeddings = embs if self.embeddings is None else np.vstack([self.embeddings, embs])
        self.chunks.extend(chunks)

        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn.fit(self.embeddings)

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        if self.nn is None or self.embeddings is None:
            raise RuntimeError("Index is empty. Add chunks first.")

        q = self.embedder.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)

        distances, indices = self.nn.kneighbors(q, n_neighbors=min(top_k, len(self.chunks)))
        sims = 1.0 - distances[0]

        results: List[Tuple[Chunk, float]] = []
        for idx, sim in zip(indices[0], sims):
            results.append((self.chunks[int(idx)], float(sim)))
        return results


def build_prompt(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    ctx_blocks = []
    for i, (chunk, score) in enumerate(retrieved, 1):
        meta = chunk.meta
        header = (
            f"[{i}] score={score:.3f} "
            f"form={meta.get('form')} date={meta.get('filing_date')} "
            f"doc={meta.get('resolved_doc')} "
            f"source={meta.get('source')}"
        )
        ctx_blocks.append(header + "\n" + chunk.text)

    context = "\n\n---\n\n".join(ctx_blocks)

    return f"""You are a helpful analyst. Answer the user's question using ONLY the context provided.
If the answer is not in the context, say you don't know.
Cite sources like [1], [2] referencing the context blocks.

Question: {question}

Context:
{context}
"""


# -----------------------------
# Ollama generation (local)
# -----------------------------
def call_ollama(prompt: str, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL) -> str:
    """
    Calls local Ollama server. No API keys.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # You can tweak temperature, top_p, etc:
        "options": {"temperature": 0.2},
    }

    try:
        r = requests.post(url, json=payload, timeout=300)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is installed and running.\n"
            "Try: ollama run llama3.1:8b \"hello\""
        ) from e

    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")

    data = r.json()
    return data.get("response", "").strip()

#test
# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cik = os.environ.get("CIK", "0000320193")
    form = os.environ.get("FORM", "10-K")
    question = os.environ.get("QUESTION", "What does the company say are its main risk factors related to supply chain?")

    submissions = get_company_submissions(cik)
    accession_nodash, primary_doc, filing_date = pick_recent_filing(submissions, form_type=form)

    print(f"Using CIK={cik} form={form} filing_date={filing_date} primary_doc={primary_doc}")

    filing_text, resolved_doc = download_filing_text_resolved(cik, accession_nodash, primary_doc)

    raw_chunks = chunk_text(filing_text, chunk_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP)
    if not raw_chunks:
        raise RuntimeError("No text extracted from filing (empty after HTML->text).")

    cik_int = str(int(cik))
    source = f"sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{resolved_doc}"

    chunks: List[Chunk] = [
        Chunk(
            text=ch,
            meta={
                "cik": cik,
                "form": form,
                "filing_date": filing_date,
                "primary_doc": primary_doc,
                "resolved_doc": resolved_doc,
                "source": source,
            },
        )
        for ch in raw_chunks
        if len(ch.strip()) >= 200
    ]

    print(f"Extracted {len(filing_text):,} chars, created {len(chunks)} chunks.")

    rag = SklearnVectorIndex()
    rag.add(chunks)

    retrieved = rag.search(question, top_k=TOP_K)

    print("\nTop retrieved chunks (preview):")
    for i, (c, s) in enumerate(retrieved, 1):
        preview = c.text[:260].replace("\n", " ")
        print(f"{i}. sim={s:.3f} doc={c.meta.get('resolved_doc')} date={c.meta.get('filing_date')}")
        print(f"   {preview}...\n")

    prompt = build_prompt(question, retrieved)

    if USE_OLLAMA:
        answer = call_ollama(prompt)
        print("\n--- Answer (Ollama, grounded) ---\n")
        print(answer)
    else:
        print("\nOllama disabled (USE_OLLAMA=0). Prompt preview:\n")
        print(prompt[:2500])


if __name__ == "__main__":
    main()

