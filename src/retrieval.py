'''
Retrieves and reconstructs candidate documents relative to a query.

Functions:
- tokenize(text: str) -> list:
    Purpose: Tokenizes a string into a list of lowercase alphanumeric words.
- compute_bm25_scores(query: str, corpus: list, k1: float = 1.5, b: float = 0.75) -> list:
    Purpose: Computes BM25 scores for a list of document dicts.
- parse_srse_row(srse_text: str) -> tuple:
    Purpose: Parses an SRSE formatted row string into headers and values.
- reconstruct_table(table_id: str, table_row_index: list) -> str:
    Purpose: Reconstructs the full table in Markdown format using all row chunks matching table_id.
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
# Import Libraries
import re
from rank_bm25 import BM25Okapi
from tabulate import tabulate

# Tokenize text into words
def tokenize(text: str) -> list:
    """
    Tokenizes a string into a list of lowercase alphanumeric words.
    """
    # Check if the text is empty
    if not text:
        return []
    # Return the text in lowercase
    return re.findall(r'\b\w+\b', text.lower())

# Implement BM25 algorithm to compute scores
def compute_bm25_scores(query: str, corpus: list, k1: float = 1.5, b: float = 0.75) -> list:
    """
    Computes BM25 scores for a list of document dicts.
    Each document dict must have a 'text' key.
    
    **Args**:
        query (str): The search query.
        corpus (list): List of document dicts.
        k1 (float): BM25 parameter. It controls how quickly the score "saturates" 
          as a word appears multiple times in a document
        b (float): BM25 parameter. It controls how much weight the document length 
          has on the final score.

    **Returns**: list of tuples (score, index).
    """
    # Check if the corpus is empty
    if not corpus:
        return []
    
    # Tokenize the query
    query_terms = tokenize(query)
    
    # If query has no words, return 0 scores for all documents
    if not query_terms:
        return [(0.0, idx) for idx in range(len(corpus))]
    
    # Pre-tokenize all documents
    doc_tokens = [tokenize(doc.get("text", "")) for doc in corpus]
    
    # Initialize BM25Okapi from rank_bm25
    bm25 = BM25Okapi(doc_tokens, k1=k1, b=b)
    
    # Calculate scores
    doc_scores = bm25.get_scores(query_terms)
    
    # Return scores as tuple
    return [(float(score), idx) for idx, score in enumerate(doc_scores)]

def parse_srse_row(srse_text: str) -> tuple:
    """
    Parses an SRSE formatted row string.
    Expected format: "Table: <table_id> | Header1: Value1 | Header2: Value2 | ..."

    **Args**:
        srse_text (str): The SRSE formatted row string.

    **Returns**:
        tuple: (headers, values)
    """
    # Split the SRSE formatted row string into parts
    parts = srse_text.split(" | ")
    
    # Initialize headers and values
    headers = []
    values = []
    
    # Loop through the parts and extract headers and values
    # Skip the first part "Table: <table_id>"
    for part in parts[1:]:
        # Split the part into header and value
        if ":" in part:
            h, v = part.split(":", 1)
            headers.append(h.strip())
            values.append(v.strip())
        # If the part does not have a colon, it is a value
        else:
            headers.append("Column")
            values.append(part.strip())
    
    # Return headers and values
    return headers, values

def reconstruct_table(table_id: str, table_row_index: list) -> str:
    """
    Reconstructs the full table in Markdown format using all row chunks matching table_id.

    Args:
        table_id (str): The ID of the table to reconstruct.
        table_row_index (list): List of table row document chunks.
    
    Returns:
        str: The full table in Markdown format.
    """
    # Filter rows matching table_id
    rows = [r for r in table_row_index if r.get("metadata", {}).get("table_id") == table_id]
    # If no rows found, return empty string
    if not rows:
        return ""
    
    # Sort rows by row_index
    rows.sort(key=lambda r: r.get("metadata", {}).get("row_index", 0))
    
    # Initialize headers and values
    table_data = []
    headers = []
    
    # Loop through rows and extract headers and values
    for row in rows:
        r_headers, r_values = parse_srse_row(row.get("text", ""))
        
        # Set headers to first row with headers
        if r_headers and not headers:
            headers = r_headers
        
        # Append values
        table_data.append(r_values)
    
    # Fallback to newline-separated text if headers couldn't be parsed
    if not headers:
        return "\n".join(r.get("text", "") for r in rows)
        
    # Return Markdown Table
    # Use github format since we dont care about column alignment
    return tabulate(table_data, headers=headers, tablefmt="github")

def retrieve_naive(query: str, indexed_chunks: list, top_k: int = 5) -> list:
    """
    Executes a standard nearest-neighbor semantic search using BM25 
    over flat naive chunks.
    
    Args:
        query (str): The search query.
        indexed_chunks (list): List of naive document chunks.
        top_k (int): Number of chunks to retrieve.
        
    Returns:
        list: Top k retrieved document chunks.
    """
    # Check if the indexed chunks list is empty
    if not indexed_chunks:
        return []

    # Compute BM25 scores
    scores = compute_bm25_scores(query, indexed_chunks)

    # Sort scores in descending order
    scores.sort(key=lambda x: x[0], reverse=True)

    # Return top k retrieved document chunks
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
    # Combine text_index and table_row_index
    combined_corpus = text_index + table_row_index

    # Check if the combined corpus is empty
    if not combined_corpus:
        return []

    # Compute BM25 scores
    scores = compute_bm25_scores(query, combined_corpus)

    # Sort scores in descending order
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Initialize retrieved_results and reconstructed_tables
    retrieved_results = []
    reconstructed_tables = set()
    
    # Loop through scores and retrieve top k chunks
    for score, idx in scores:
        # Check if the retrieved_results list has reached the top_k limit
        if len(retrieved_results) >= top_k:
            break
            
        # Get the current chunk
        chunk = combined_corpus[idx]
        
        # Get the metadata
        metadata = chunk.get("metadata", {})
        
        # Get the source
        source = metadata.get("source")
        
        # Check if the source is table_aware_row
        if source == "table_aware_row":
            table_id = metadata.get("table_id")
            
            # Check if the table_id is not found
            if not table_id:
                retrieved_results.append(chunk)
                continue
                
            # Check if the table_id is not in reconstructed_tables
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
