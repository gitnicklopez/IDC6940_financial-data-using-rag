'''
Retrieves and reconstructs candidate documents relative to a query.

Functions:
- retrieve_naive(query: str, indexed_chunks: list[dict], top_k: int = 5) -> list[dict]:
    Purpose: Executes a standard nearest-neighbor semantic search (using lightweight TF-IDF, embeddings, 
    or string overlap) over the flat naive chunks.
- retrieve_table_aware(query: str, text_index: list[dict], table_row_index: list[dict], top_k: int = 5) -> list[dict]:
    Purpose: Executes dual-channel retrieval:
        1. Channel I: Semantic search across standard narrative prose.
        2. Channel II: Match search queries to specific table rows (SRSE).
        3. Reconstruction & Fusion: If a table row is retrieved, it automatically reconstructs the immediate table block
           Headers + neighboring rows) via metadata keys, providing structured Markdown tables rather than floating,
           disconnected values.
'''

import math
import re

def tokenize(text: str) -> list:
    """
    Tokenizes a string into a list of lowercase alphanumeric words.
    """
    if not text:
        return []
    return re.findall(r'\b\w+\b', text.lower())

def compute_bm25_scores(query: str, corpus: list, k1: float = 1.5, b: float = 0.75) -> list:
    """
    Computes BM25 scores for a list of document dicts.
    Each document dict must have a 'text' key.
    
    Returns a list of tuples (score, index).
    """
    if not corpus:
        return []
    
    query_terms = tokenize(query)
    if not query_terms:
        # If query has no words, return 0 scores for all documents
        return [(0.0, idx) for idx in range(len(corpus))]
    
    # Pre-tokenize all documents
    doc_tokens = [tokenize(doc.get("text", "")) for doc in corpus]
    doc_lens = [len(tokens) for tokens in doc_tokens]
    avg_len = sum(doc_lens) / len(corpus) if corpus else 1.0
    N = len(corpus)
    
    # Calculate Document Frequencies (DF) for query terms
    df = {}
    for term in set(query_terms):
        df[term] = sum(1 for tokens in doc_tokens if term in tokens)
        
    scores = []
    for idx, doc in enumerate(corpus):
        tokens = doc_tokens[idx]
        doc_len = doc_lens[idx]
        
        # Term frequencies for this document
        tf_map = {}
        for token in tokens:
            tf_map[token] = tf_map.get(token, 0) + 1
            
        score = 0.0
        for term in query_terms:
            if term not in tf_map:
                continue
            n_qi = df[term]
            
            # BM25 IDF formula with smoothing to avoid negative IDF values
            idf = math.log((N - n_qi + 0.5) / (n_qi + 0.5) + 1.0)
            tf = tf_map[term]
            denom = tf + k1 * (1.0 - b + b * (doc_len / avg_len))
            score += idf * (tf * (k1 + 1.0)) / denom
            
        scores.append((score, idx))
        
    return scores

def parse_srse_row(srse_text: str) -> tuple:
    """
    Parses an SRSE formatted row string.
    Expected format: "Table: <table_id> | Header1: Value1 | Header2: Value2 | ..."
    """
    parts = srse_text.split(" | ")
    headers = []
    values = []
    # Skip the first part: "Table: <table_id>"
    for part in parts[1:]:
        if ":" in part:
            h, v = part.split(":", 1)
            headers.append(h.strip())
            values.append(v.strip())
        else:
            headers.append("Column")
            values.append(part.strip())
    return headers, values

def reconstruct_table(table_id: str, table_row_index: list) -> str:
    """
    Reconstructs the full table in Markdown format using all row chunks matching table_id.
    """
    # Filter rows matching table_id
    rows = [r for r in table_row_index if r.get("metadata", {}).get("table_id") == table_id]
    if not rows:
        return ""
    
    # Sort rows by row_index
    rows.sort(key=lambda r: r.get("metadata", {}).get("row_index", 0))
    
    table_data = []
    headers = []
    
    for row in rows:
        r_headers, r_values = parse_srse_row(row.get("text", ""))
        if r_headers and not headers:
            headers = r_headers
        table_data.append(r_values)
        
    if not headers:
        # Fallback to newline-separated text if headers couldn't be parsed
        return "\n".join(r.get("text", "") for r in rows)
        
    # Format as Markdown Table
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    markdown_lines = [header_line, separator_line]
    for row_values in table_data:
        # Ensure values list matches headers length
        padded_values = row_values + [""] * (len(headers) - len(row_values))
        padded_values = padded_values[:len(headers)]
        markdown_lines.append("| " + " | ".join(padded_values) + " |")
        
    return "\n".join(markdown_lines)

def retrieve_naive(query: str, indexed_chunks: list, top_k: int = 5) -> list:
    """
    Executes a standard nearest-neighbor semantic search (using lightweight BM25)
    over flat naive chunks.
    
    Args:
        query (str): The search query.
        indexed_chunks (list): List of naive document chunks.
        top_k (int): Number of chunks to retrieve.
        
    Returns:
        list: Top k retrieved document chunks.
    """
    if not indexed_chunks:
        return []
    scores = compute_bm25_scores(query, indexed_chunks)
    scores.sort(key=lambda x: x[0], reverse=True)
    return [indexed_chunks[idx] for _, idx in scores[:top_k]]

def retrieve_table_aware(query: str, text_index: list, table_row_index: list, top_k: int = 5) -> list:
    """
    Executes dual-channel retrieval:
    1. Channel I: Semantic search across standard narrative prose (text_index).
    2. Channel II: Match search queries to specific table rows (table_row_index).
    3. Reconstruction & Fusion: If a table row is retrieved, it automatically reconstructs 
       the immediate table block (headers + neighboring rows) via metadata keys,
       providing structured Markdown tables.
       
    Args:
        query (str): The search query.
        text_index (list): List of narrative prose document chunks.
        table_row_index (list): List of table row document chunks.
        top_k (int): Number of unique contexts to retrieve.
        
    Returns:
        list: Top k retrieved and fused document chunks.
    """
    combined_corpus = text_index + table_row_index
    if not combined_corpus:
        return []
        
    scores = compute_bm25_scores(query, combined_corpus)
    scores.sort(key=lambda x: x[0], reverse=True)
    
    retrieved_results = []
    reconstructed_tables = set()
    
    for score, idx in scores:
        if len(retrieved_results) >= top_k:
            break
            
        chunk = combined_corpus[idx]
        metadata = chunk.get("metadata", {})
        source = metadata.get("source")
        
        if source == "table_aware_row":
            table_id = metadata.get("table_id")
            if not table_id:
                retrieved_results.append(chunk)
                continue
                
            if table_id not in reconstructed_tables:
                reconstructed_tables.add(table_id)
                # Reconstruct full table block and create unified context chunk
                reconstructed_text = reconstruct_table(table_id, table_row_index)
                reconstructed_chunk = {
                    "text": reconstructed_text,
                    "metadata": {
                        "source": "reconstructed_table",
                        "table_id": table_id,
                        "page": metadata.get("page", "unknown")
                    }
                }
                retrieved_results.append(reconstructed_chunk)
            # Duplicate row references are skipped because their table is already fused
        else:
            # Prose narrative chunk
            retrieved_results.append(chunk)
            
    return retrieved_results
