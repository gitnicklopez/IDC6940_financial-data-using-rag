"""
This module implements the RAG pipeline for ingesting and indexing PDF documents.

Functions:
- run_pipeline(pdf_dir: str) -> dict:
    - Purpose: Ingests and indexes all PDF documents in the specified directory using both naive and table-aware pipelines.
    - Args:
        pdf_dir (str): Directory containing the PDF files.
    - Returns:
        dict: A dictionary containing the naive, text, and table row indices.
"""
import os

from src.ingestion import parse_document_linear, parse_document_table_aware
from src.indexing import index_naive_chunks, index_table_aware_rows, export_indices_to_json

def run_pipeline(pdf_dir: str) -> dict:
    """
    Ingests and indexes all PDF documents in the specified directory.
    
    Args:
        pdf_dir (str): Directory containing the PDF files.
    
    Returns:
        dict: A dictionary containing the naive, text, and table row indices.
    """
    # Check if the directory exists
    if not os.path.isdir(pdf_dir):
        raise ValueError(f"The directory {pdf_dir} does not exist.")

    # Initialize indices
    naive_index = []
    text_index = []
    table_row_index = []
    
    # Process each PDF file in the directory for each RAG pipeline
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            
            # --- Naive Pipeline ---
            linear_data = parse_document_linear(pdf_path)
            naive_chunks = index_naive_chunks(linear_data.get("text", ""))
            for chunk in naive_chunks:
                chunk["metadata"]["filename"] = filename
            naive_index.extend(naive_chunks)
            
            # --- Table-Aware Pipeline ---
            parsed_data = parse_document_table_aware(pdf_path, table_extractor="camelot")
            indexed_data = index_table_aware_rows(parsed_data)
            
            text_chunks = indexed_data.get("text", [])
            table_chunks = indexed_data.get("tables", [])
            
            for chunk in text_chunks:
                chunk["metadata"]["filename"] = filename
            for chunk in table_chunks:
                chunk["metadata"]["filename"] = filename
                
            text_index.extend(text_chunks)
            table_row_index.extend(table_chunks)

    # Export the indices to JSON files
    export_indices_to_json(naive_index, text_index, table_row_index)

    return {
        "naive_index": naive_index,
        "text_index": text_index,
        "table_row_index": table_row_index
    }


