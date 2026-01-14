"""Text chunking utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sec_rag.config import CHUNK_CHARS, CHUNK_OVERLAP, MIN_CHUNK_LENGTH
from sec_rag.utils import load_json, save_json


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    meta: Dict[str, Any]
    chunk_id: Optional[str] = None


def chunk_text(
    text: str,
    chunk_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
    min_length: int = MIN_CHUNK_LENGTH
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_chars: Target characters per chunk
        overlap: Overlap between chunks
        min_length: Minimum chunk length to keep
        
    Returns:
        List of chunk strings
    """
    text = text.strip()
    if not text:
        return []
    
    chunks: List[str] = []
    start = 0
    n = len(text)
    
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        
        if len(chunk) >= min_length:
            chunks.append(chunk)
        
        if end == n:
            break
        
        start = max(0, end - overlap)
    
    return chunks


def create_chunks_from_text(
    text: str,
    metadata: Dict[str, Any],
    chunk_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP
) -> List[Chunk]:
    """
    Create Chunk objects from text with metadata.
    
    Args:
        text: Text to chunk
        metadata: Base metadata to attach to each chunk
        chunk_chars: Target characters per chunk
        overlap: Overlap between chunks
        
    Returns:
        List of Chunk objects
    """
    raw_chunks = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
    
    chunks: List[Chunk] = []
    for i, chunk_text_content in enumerate(raw_chunks):
        chunk_meta = metadata.copy()
        chunk_meta["chunk_id"] = f"{metadata.get('ticker', '')}_{metadata.get('accession_nodash', '')}_{i}"
        chunk_meta["chunk_index"] = i
        
        chunks.append(Chunk(
            text=chunk_text_content,
            meta=chunk_meta,
            chunk_id=chunk_meta["chunk_id"]
        ))
    
    return chunks


def save_chunks(chunks: List[Chunk], filepath: Path) -> None:
    """Save chunks to JSON file."""
    data = [
        {
            "text": c.text,
            "meta": c.meta,
            "chunk_id": c.chunk_id
        }
        for c in chunks
    ]
    save_json(data, filepath)


def load_chunks(filepath: Path) -> List[Chunk]:
    """Load chunks from JSON file."""
    data = load_json(filepath)
    return [
        Chunk(
            text=item["text"],
            meta=item["meta"],
            chunk_id=item.get("chunk_id")
        )
        for item in data
    ]

