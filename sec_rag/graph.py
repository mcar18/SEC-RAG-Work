"""GraphRAG: Entity graph construction and graph-aware retrieval."""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from sec_rag.chunking import Chunk
from sec_rag.config import GRAPH_ALPHA, GRAPHS_DIR
from sec_rag.utils import load_json, save_json


# Simple entity extraction patterns (fallback if spaCy not available)
COUNTRY_PATTERNS = [
    r"\b(?:United States|China|India|Japan|Germany|United Kingdom|France|Italy|Brazil|Russia|Canada|South Korea|Mexico|Australia|Spain)\b",
    r"\b(?:US|USA|UK|EU|U\.S\.|U\.K\.)\b",
]

ORG_PATTERNS = [
    r"\b[A-Z][a-z]+ (?:Inc|Corp|Corporation|LLC|Ltd|Limited|Company|Co\.|Technologies|Systems|Group)\b",
    r"\b(?:SEC|FDA|FTC|DOJ|EU|WTO|UN)\b",
]

PRODUCT_TECH_PATTERNS = [
    r"\b(?:AI|artificial intelligence|machine learning|cloud computing|blockchain|cryptocurrency|IoT|internet of things)\b",
    r"\b(?:iPhone|iPad|Mac|Windows|Azure|AWS|GCP|TensorFlow|PyTorch)\b",
]

REGULATION_PATTERNS = [
    r"\b(?:GDPR|CCPA|HIPAA|SOX|Dodd-Frank|SEC Rule|FDA regulation)\b",
    r"\b(?:regulation|compliance|regulatory|legislation)\b",
]


def extract_entities_simple(text: str) -> Set[str]:
    """
    Extract entities using simple regex patterns (fallback method).
    
    Args:
        text: Text to extract entities from
        
    Returns:
        Set of entity strings
    """
    entities = set()
    
    # Try to extract countries
    for pattern in COUNTRY_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities.update(m.lower() for m in matches)
    
    # Try to extract organizations
    for pattern in ORG_PATTERNS:
        matches = re.findall(pattern, text)
        entities.update(m.lower() for m in matches)
    
    # Try to extract products/tech
    for pattern in PRODUCT_TECH_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities.update(m.lower() for m in matches)
    
    # Try to extract regulations
    for pattern in REGULATION_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities.update(m.lower() for m in matches)
    
    return entities


def build_entity_graph(chunks: List[Chunk]) -> nx.Graph:
    """
    Build an entity graph from chunks.
    
    Nodes are entities, edges connect entities that co-occur in the same chunk.
    Edge weights = co-occurrence count.
    
    Args:
        chunks: List of chunks
        
    Returns:
        NetworkX graph
    """
    G = nx.Graph()
    co_occurrence = defaultdict(int)
    
    for chunk in chunks:
        entities = extract_entities_simple(chunk.text)
        entities = {e for e in entities if len(e) > 2}  # Filter very short entities
        
        # Add nodes
        for entity in entities:
            G.add_node(entity)
        
        # Add edges for co-occurring entities
        entities_list = list(entities)
        for i, e1 in enumerate(entities_list):
            for e2 in entities_list[i+1:]:
                if e1 != e2:
                    pair = tuple(sorted([e1, e2]))
                    co_occurrence[pair] += 1
    
    # Add edges with weights
    for (e1, e2), weight in co_occurrence.items():
        if G.has_edge(e1, e2):
            G[e1][e2]["weight"] += weight
        else:
            G.add_edge(e1, e2, weight=weight)
    
    return G


def extract_query_entities(query: str) -> Set[str]:
    """Extract entities from a query."""
    return extract_entities_simple(query)


def graph_boost_score(
    query_entities: Set[str],
    chunk: Chunk,
    graph: nx.Graph,
    alpha: float = GRAPH_ALPHA
) -> float:
    """
    Calculate graph boost score for a chunk.
    
    Args:
        query_entities: Entities found in query
        chunk: Chunk to score
        graph: Entity graph
        alpha: Weight for cosine similarity vs graph boost
        
    Returns:
        Graph boost score (0-1)
    """
    chunk_entities = extract_entities_simple(chunk.text)
    
    if not query_entities or not chunk_entities:
        return 0.0
    
    # Find entities in chunk that are connected to query entities
    boost = 0.0
    for q_entity in query_entities:
        if q_entity not in graph:
            continue
        
        # Get neighbors of query entity
        neighbors = set(graph.neighbors(q_entity))
        
        # Count how many chunk entities are neighbors
        overlap = len(neighbors & chunk_entities)
        if overlap > 0:
            # Boost based on edge weights
            total_weight = sum(graph[q_entity][n].get("weight", 1) for n in neighbors & chunk_entities)
            boost += total_weight / (len(neighbors) + 1)  # Normalize
    
    # Normalize boost to 0-1 range
    max_possible_boost = len(query_entities) * 2.0  # Rough normalization
    normalized_boost = min(1.0, boost / max_possible_boost) if max_possible_boost > 0 else 0.0
    
    return normalized_boost


def graph_aware_search(
    query: str,
    index,
    graph: nx.Graph,
    top_k: int = 6,
    alpha: float = GRAPH_ALPHA,
    filter_meta: Optional[dict] = None
) -> List[Tuple[Chunk, float]]:
    """
    Perform graph-aware retrieval.
    
    Args:
        query: Query text
        index: Vector index
        graph: Entity graph
        top_k: Number of results
        alpha: Weight for cosine similarity (1-alpha for graph boost)
        filter_meta: Optional metadata filter
        
    Returns:
        List of (Chunk, final_score) tuples
    """
    # Get base retrieval results
    base_results = index.search(query, top_k=top_k * 3, filter_meta=filter_meta)
    
    # Extract query entities
    query_entities = extract_query_entities(query)
    
    # Combine cosine similarity with graph boost
    final_results = []
    for chunk, cosine_sim in base_results:
        graph_boost = graph_boost_score(query_entities, chunk, graph, alpha=alpha)
        final_score = alpha * cosine_sim + (1 - alpha) * graph_boost
        final_results.append((chunk, final_score))
    
    # Sort by final score and return top-k
    final_results.sort(key=lambda x: x[1], reverse=True)
    return final_results[:top_k]


def build_graphs_for_filings(
    filings_data: Dict[str, Dict[int, Tuple]],
    output_dir: Optional[Path] = None
) -> Dict[str, nx.Graph]:
    """
    Build entity graphs for all filings.
    
    Args:
        filings_data: Nested dict {ticker: {year: (metadata, chunks, index)}}
        output_dir: Optional output directory
        
    Returns:
        Dict of {filing_key: graph}
    """
    if output_dir is None:
        output_dir = GRAPHS_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graphs = {}
    
    for ticker, year_data in filings_data.items():
        for year, (metadata, chunks, index) in year_data.items():
            print(f"Building graph for {ticker} {year}...")
            
            graph = build_entity_graph(chunks)
            graphs[f"{ticker}_{year}"] = graph
            
            # Save graph
            graph_path = output_dir / f"{ticker}_{metadata.cik}_{metadata.form}_{year}_{metadata.accession_nodash}.json"
            
            # Convert to JSON-serializable format
            graph_data = {
                "nodes": list(graph.nodes()),
                "edges": [
                    {"source": u, "target": v, "weight": d.get("weight", 1)}
                    for u, v, d in graph.edges(data=True)
                ]
            }
            save_json(graph_data, graph_path)
            print(f"Saved graph to {graph_path} ({len(graph.nodes())} nodes, {len(graph.edges())} edges)")
    
    return graphs

