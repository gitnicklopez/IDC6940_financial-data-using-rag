'''
Handles document segmenting, metadata enrichment, and vector representation mappings.

Funtions:
- index_naive_chunks(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[dict]:
    - Purpose: Partitions linearized text streams into fixed-size chunks ignoring structural boundaries.
- index_table_aware_rows(parsed_data: dict) -> dict[str, list]:
    - Purpose: Separates indexing into two categories:
        1. Chunks continuous narrative prose standardly.
        2. Implements **Single-Row Single-Embedding (SRSE)** by converting each isolated table row into a structured Markdown line enriched with header metadata, index references, and table coordinates.
- _format_row_as_srse(row_cells: list, headers: list, table_id: str) -> str
    - Purpose: Formats cell sequences into metadata-rich strings to bind column definitions to values.
'''
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os

def index_naive_chunks(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[dict]:
    '''
    Partitions linearized text streams into fixed-size chunks ignoring structural boundaries.
    
    Args:
        text (str): Linearized text content.
        chunk_size (int): Maximum number of tokens per chunk.
        chunk_overlap (int): Number of overlapping tokens between consecutive chunks.

    Returns:
        list[dict]: List of indexed chunks.
    '''
    # Create a splitter for text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(x.split()),
        separators=[" ", ""]
    )
    
    # Split text into tokens
    split_texts = splitter.split_text(text)
    
    # Add chunks to list
    chunks = []
    for chunk_text in split_texts:
        chunks.append({"text": chunk_text, "metadata": {"source": "naive", "page": "unknown"}})
    
    return chunks

def index_table_aware_rows(parsed_data: dict, chunk_size: int = 512, chunk_overlap: int = 50) -> dict:
    '''
    Separates indexing into two categories:
    1. Chunks continuous narrative prose standardly.
    2. Implements **Single-Row Single-Embedding (SRSE)** by converting each isolated table row into a structured 
    Markdown line enriched with header metadata, index references, and table coordinates.
    
    Args:
        parsed_data (dict): Dictionary containing 'text' and 'tables'.
        chunk_size (int): Maximum number of tokens per chunk.
        chunk_overlap (int): Number of overlapping tokens between consecutive chunks.

    Returns:
        dict: Dictionary containing 'text', 'tables', and 'metadata'.
    '''
    # Initialize metadata variables
    table_aware_text = []
    
    # Define splitter for words
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(x.split()),
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Chunk continuous narrative prose standardly
    for text_block in parsed_data.get("text", []):
        lines = text_block.strip().split("\n")
        if not lines:
            continue
            
        header_line = lines[0]

        # Extract page number (e.g. from "--- Page 1 (Text) ---")
        page_num = "unknown"
        if "Page" in header_line:
            try:
                page_num = header_line.split("Page")[1].split("(")[0].strip()
            except Exception:
                pass

        # Get the content of the block (excluding the header)
        block_content = "\n".join(lines[1:])
        
        split_texts = splitter.split_text(block_content)
        for chunk_text in split_texts:
            table_aware_text.append({
                "text": chunk_text,
                "metadata": {"source": "table_aware_text", "page": page_num}
            })

    # Extract and format table rows using SRSE
    table_chunks = []
    for table_str in parsed_data.get("tables", []):
        lines = table_str.strip().split("\n")

        # Skip empty or single-line tables
        if len(lines) < 2:
            continue
            
        # Parse table ID/metadata from header (e.g. "--- Page 35 Table 1 ---")
        table_id = "unknown_table"
        page_num = "unknown"
        header_line = lines[0]
        if "Page" in header_line:
            # Try to extract table ID and page number
            try:
                parts = header_line.replace("-", "").strip().split()
                # Expected parts: ["Page", "35", "Table", "1"]
                if "Page" in parts and "Table" in parts:
                    p_idx = parts.index("Page")
                    t_idx = parts.index("Table")
                    page_num = parts[p_idx + 1]
                    table_num = parts[t_idx + 1]
                    table_id = f"Page_{page_num}_Table_{table_num}"
            # Fallback: extract table id and page number
            except Exception:
                table_id = header_line.replace("-", "").strip().replace(" ", "_")
                
        # The first row of table data is the header columns
        headers = [c.strip() for c in lines[1].split("|")]
        
        # Process data rows
        for r_idx, line in enumerate(lines[2:]):
            row_cells = [c.strip() for c in line.split("|")]
            srse_str = _format_row_as_srse(row_cells, headers, table_id)
            table_chunks.append({
                "text": srse_str,
                "metadata": {
                    "source": "table_aware_row",
                    "table_id": table_id,
                    "page": page_num,
                    "row_index": r_idx + 1
                }
            })

    return {
        "text": table_aware_text, 
        "tables": table_chunks, 
        "metadata": parsed_data.get('metadata', {})
    }

def _format_row_as_srse(row_cells: list, headers: list, table_id: str) -> str:
    '''
    Formats cell sequences into metadata-rich strings to bind column definitions to values.
    
    Args:
        row_cells (list): Cells in the row.
        headers (list): Header column names.
        table_id (str): ID identifying the table.

    Returns:
        str: Formatted metadata-rich string.
    '''
    # Format the row as a metadata-rich string
    parts = [f"Table: {table_id}"]

    # Add each cell as a key-value pair
    for i, cell in enumerate(row_cells):
        header_name = headers[i] if i < len(headers) else f"Col_{i+1}"
        parts.append(f"{header_name}: {cell}")
    
    # Return formatted metadata-rich string
    return " | ".join(parts)

def export_indices_to_json(naive_index: list, text_index: list, table_row_index: list, output_dir: str = "data/indexed"):
    '''
    Exports the generated indices to JSON files.
    
    Args:
        naive_index (list): List of naive index chunks.
        text_index (list): List of text chunks for the table-aware pipeline.
        table_row_index (list): List of table row chunks (SRSE) for the table-aware pipeline.
        output_dir (str): Directory to save the JSON files.
    '''
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Export naive index
    naive_path = os.path.join(output_dir, "naive.json")
    with open(naive_path, 'w', encoding='utf-8') as f:
        json.dump(naive_index, f, indent=4, ensure_ascii=False)
        
    # Export table-aware text (prose) index
    text_path = os.path.join(output_dir, "table-aware_prose.json")
    with open(text_path, 'w', encoding='utf-8') as f:
        json.dump(text_index, f, indent=4, ensure_ascii=False)
        
    # Export table-aware SRSE index
    table_path = os.path.join(output_dir, "table-aware_SRSE.json")
    with open(table_path, 'w', encoding='utf-8') as f:
        json.dump(table_row_index, f, indent=4, ensure_ascii=False)
