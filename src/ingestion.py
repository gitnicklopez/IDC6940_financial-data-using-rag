'''
Implements PDF text and tabular extraction methods using `pdfplumber`.

Funtions:
- parse_document_linear(pdf_path: str) -> str:
    Purpose:
      Recreates Naive RAG ingestion. Extracts all pages as a singular, one-dimensional stream of characters, 
      flattening tables into continuous strings.
- parse_document_table_aware(pdf_path: str) -> dict:
    Purpose:
      Separates narrative text from tabular structures. Returns a dictionary containing lists of extracted text blocks
      and structured tables (retaining tabular cells, column names, and vertical layout boundaries)
'''

import pdfplumber
from typing import List, Dict, Any

def parse_document_linear(pdf_path: str) -> Dict[str, Any]:
    """
    Reads the entire PDF and returns a dictionary with continuous text string
    and extraction metadata. Tables are flattened into the stream without structural preservation.

    **Args**:
        pdf_path (str): Path to the PDF document.

    **Returns**:
        Dict[str, Any]: Dictionary containing 'text' and 'metadata'.
    """
    # Initialize metadata variables
    full_text = ""
    num_pages = 0
    num_text_blocks = 0
    
    # Open PDF file for reading
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Set number of pages
            num_pages = len(pdf.pages)
            # Iterate through each page in the PDF
            for page in pdf.pages:
                # Extract text from the current page
                page_text = page.extract_text(x_tolerance=3) # Added x_tolerance=3 for word spacing
                if page_text:
                    full_text += page_text + "\n--- Page Break ---\n" # Added page break between pages
                    num_text_blocks += 1
    
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    # Return full text and metadata    
    return {
        "text": full_text,
        "metadata": {
            "num_documents": 1 if num_pages > 0 else 0,
            "num_pages": num_pages,
            "num_text_blocks": num_text_blocks,
            "num_tables": 0,
            "num_rows": 0
        }
    }


def parse_document_table_aware(pdf_path: str) -> Dict[str, Any]:
    """
    Extracts text and tables separately, preserving table structure.
    Returns a dictionary with 'text' blocks, 'tables' (as formatted strings),
    and extraction metadata.

    **Args**:
        pdf_path (str): Path to the PDF document.

    **Returns**:
        Dict[str, Any]: Dictionary containing 'text', 'tables', and 'metadata'.
    """
    # Initialize metadata variables
    data: Dict[str, Any] = {
        "text": [],
        "tables": [],
        "metadata": {
            "num_documents": 0,
            "num_pages": 0,
            "num_text_blocks": 0,
            "num_tables": 0,
            "num_rows": 0
        }
    }
    
    # Open PDF file for reading
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Set number of pages
            num_pages = len(pdf.pages)
            data["metadata"]["num_pages"] = num_pages
            # Set number of documents
            if num_pages > 0:
                data["metadata"]["num_documents"] = 1
                
            for i, page in enumerate(pdf.pages):
                # Extract text blocks (prose)
                page_text = page.extract_text(x_tolerance=3)
                if page_text:
                    data["text"].append(f"--- Page {i+1} (Text) ---\n{page_text}")
                    data["metadata"]["num_text_blocks"] += 1
                
                # Extract tables, maintaining structure
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    data["metadata"]["num_tables"] += 1
                    data["metadata"]["num_rows"] += len(table)
                    
                    # Convert the table list-of-lists into a readable Markdown/string format
                    # Include some metadata about its location
                    header = "\n".join([" | ".join(map(str, row)) for row in table])
                    table_str = f"--- Page {i+1} Table {t_idx+1} ---\n{header}\n"
                    data["tables"].append(table_str)
                    
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        
    return data

