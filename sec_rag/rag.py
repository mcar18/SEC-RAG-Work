"""RAG (Retrieval-Augmented Generation) pipeline."""

from typing import List, Optional, Tuple

from sec_rag.chunking import Chunk
from sec_rag.index import SklearnVectorIndex
from sec_rag.ollama_client import call_ollama
from sec_rag.config import USE_OLLAMA


def build_prompt(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    Build a prompt for LLM with retrieved context.
    
    Args:
        question: User question
        retrieved: List of (Chunk, similarity_score) tuples
        
    Returns:
        Formatted prompt string
    """
    ctx_blocks = []
    for i, (chunk, score) in enumerate(retrieved, 1):
        meta = chunk.meta
        header = (
            f"[{i}] score={score:.3f} "
            f"ticker={meta.get('ticker', 'N/A')} "
            f"form={meta.get('form', 'N/A')} "
            f"date={meta.get('filing_date', 'N/A')} "
            f"year={meta.get('year', 'N/A')} "
            f"doc={meta.get('resolved_document', 'N/A')}"
        )
        ctx_blocks.append(header + "\n" + chunk.text)
    
    context = "\n\n---\n\n".join(ctx_blocks)
    
    return f"""You are a helpful financial analyst. Answer the user's question using ONLY the context provided from SEC filings.
If the answer is not in the context, say you don't know.
Cite sources like [1], [2], [3] referencing the context blocks above.

Question: {question}

Context:
{context}

Answer:"""


def answer_question(
    question: str,
    index: SklearnVectorIndex,
    top_k: int = 6,
    filter_meta: Optional[dict] = None,
    use_ollama: bool = USE_OLLAMA
) -> Tuple[str, List[Tuple[Chunk, float]]]:
    """
    Answer a question using RAG.
    
    Args:
        question: User question
        index: Vector index
        top_k: Number of chunks to retrieve
        filter_meta: Optional metadata filter (e.g., {"ticker": "AAPL", "year": 2023})
        use_ollama: Whether to use Ollama for generation (False = retrieval only)
        
    Returns:
        Tuple of (answer_text, retrieved_chunks)
    """
    # Retrieve relevant chunks
    retrieved = index.search(question, top_k=top_k, filter_meta=filter_meta)
    
    if not retrieved:
        return "No relevant chunks found.", []
    
    # Build prompt
    prompt = build_prompt(question, retrieved)
    
    # Generate answer
    if use_ollama:
        answer = call_ollama(prompt)
    else:
        answer = "[Ollama disabled - retrieval only mode]\n\nRetrieved chunks:\n" + "\n\n".join(
            f"[{i}] {c.text[:200]}..." for i, (c, s) in enumerate(retrieved, 1)
        )
    
    return answer, retrieved

