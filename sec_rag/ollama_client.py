"""Ollama LLM client for local inference."""

from typing import Optional

import requests

from sec_rag.config import OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL


def call_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
    temperature: float = 0.2,
    timeout: int = OLLAMA_TIMEOUT
) -> str:
    """
    Call local Ollama server for text generation.
    
    Args:
        prompt: Input prompt
        model: Model name
        url: Ollama API URL
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        
    Returns:
        Generated text response
        
    Raises:
        RuntimeError: If connection fails or API returns error
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is installed and running.\n"
            f"Try: ollama run {model} \"hello\""
        ) from e
    
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")
    
    data = r.json()
    return data.get("response", "").strip()


def check_ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    try:
        base_url = OLLAMA_URL.replace("/api/generate", "")
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

