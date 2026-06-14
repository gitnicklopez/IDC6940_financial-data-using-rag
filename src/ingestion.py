'''
Implements PDF text and tabular extraction methods using `pdfplumber`.
* **`parse_document_linear(pdf_path: str) -> str`**:
  * *Purpose:* Recreates Naive RAG ingestion. Extracts all pages as a singular, one-dimensional stream of characters, flattening tables into continuous strings.
* **`parse_document_table_aware(pdf_path: str) -> dict`**:
  * *Purpose:* Separates narrative text from tabular structures. Returns a dictionary containing lists of extracted text blocks and structured tables (retaining tabular cells, column names, and vertical layout boundaries).

'''

import pdfplumber
from typing import List, Dict, Any

def parse_document_linear(pdf_path: str) -> str:
    """
    Reads the entire PDF and returns a single continuous text string.
    Tables are flattened into the stream without structural preservation.
    """
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text with layout analysis enabled to help separate columns/blocks
                # but cast to string directly for linear processing
                page_text = page.extract_text(x_tolerance=3)
                if page_text:
                    # Add a separator between pages
                    full_text += page_text + "\n--- Page Break ---\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return full_text


def parse_document_table_aware(pdf_path: str) -> Dict[str, Any]:
    """
    Extracts text and tables separately, preserving table structure.
    Returns a dictionary with 'text' blocks and 'tables' (as formatted strings).
    """
    data: Dict[str, Any] = {"text": [], "tables": []}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract text blocks (prose)
                page_text = page.extract_text(x_tolerance=3)
                if page_text:
                    data["text"].append(f"--- Page {i+1} (Text) ---\n{page_text}")
                
                # Extract tables, maintaining structure
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    # Convert the table list-of-lists into a readable Markdown/string format
                    # Include some metadata about its location
                    header = "\n".join([" | ".join(map(str, row)) for row in table])
                    table_str = f"--- Page {i+1} Table {t_idx+1} ---\n{header}\n"
                    data["tables"].append(table_str)
                    
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        
    return data
