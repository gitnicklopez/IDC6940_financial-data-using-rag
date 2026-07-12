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
import re
from typing import List, Dict, Any

def is_numeric_data(cell_str):
    clean_str = re.sub(r'[$, \(\)%]', '', str(cell_str)).strip()
    try:
        float(clean_str)
        if re.match(r'^(19|20)\d{2}(?:-\d{2})?$', clean_str):
            return False
        return True
    except ValueError:
        return False

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


def parse_document_table_aware(pdf_path: str, table_extractor: str = "pdfplumber") -> Dict[str, Any]:
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
        if table_extractor == "camelot":
            import camelot
            # Extract all tables across the document
            cam_tables = camelot.read_pdf(pdf_path, flavor='stream', pages='all')
            camelot_tables_by_page = {}
            for t in cam_tables:
                page_idx = t.page - 1
                if page_idx not in camelot_tables_by_page:
                    camelot_tables_by_page[page_idx] = []
                camelot_tables_by_page[page_idx].append(t)

        with pdfplumber.open(pdf_path) as pdf:
            # Set number of pages
            num_pages = len(pdf.pages)
            data["metadata"]["num_pages"] = num_pages
            # Set number of documents
            if num_pages > 0:
                data["metadata"]["num_documents"] = 1
                
            for i, page in enumerate(pdf.pages):
                page_height = page.height
                
                if table_extractor == "pdfplumber":
                    # Find tables on this page to exclude their boundaries from narrative prose
                    page_tables = page.find_tables()
                    
                    def not_in_table(obj):
                        if obj.get("object_type") == "char":
                            for t in page_tables:
                                bbox = t.bbox  # (x0, top, x1, bottom)
                                if bbox[0] <= obj["x0"] <= bbox[2] and bbox[1] <= obj["top"] <= bbox[3]:
                                    return False
                        return True
                        
                    # Extract tables, maintaining structure
                    tables = page.extract_tables()
                    for t_idx, table in enumerate(tables):
                        data["metadata"]["num_tables"] += 1
                        data["metadata"]["num_rows"] += len(table)
                        
                        # Clean internal newlines from cells so they don't break our line-by-line parsing later
                        clean_rows = []
                        for row in table:
                            clean_cells = [str(cell).replace('\n', ' ') if cell is not None else "" for cell in row]
                            clean_rows.append(" | ".join(clean_cells))
                        
                        header = "\n".join(clean_rows)
                        table_str = f"--- Page {i+1} Table {t_idx+1} ---\n{header}\n"
                        data["tables"].append(table_str)

                elif table_extractor == "camelot":
                    cam_page_tables = camelot_tables_by_page.get(i, [])
                    
                    # Filter out fake prose tables (must have at least one numeric data cell)
                    valid_cam_tables = []
                    for t in cam_page_tables:
                        has_numeric = False
                        for _, row in t.df.iterrows():
                            if any(is_numeric_data(cell) for cell in row.values[1:]):
                                has_numeric = True
                                break
                        if has_numeric:
                            valid_cam_tables.append(t)
                    cam_page_tables = valid_cam_tables
                    def not_in_table(obj):
                        if obj.get("object_type") == "char":
                            for t in cam_page_tables:
                                # camelot bbox: (x0, y0, x1, y1) bottom-left origin
                                cx0, cy0, cx1, cy1 = t._bbox
                                top = page_height - cy1
                                bottom = page_height - cy0
                                
                                if cx0 <= obj["x0"] <= cx1 and top <= obj["top"] <= bottom:
                                    return False
                        return True
                        
                    for t_idx, t in enumerate(cam_page_tables):
                        data["metadata"]["num_tables"] += 1
                        df = t.df
                        data["metadata"]["num_rows"] += len(df)
                        
                        data_start_idx = 0
                        for idx, row in df.iterrows():
                            # Check if any column after the first contains a number
                            if any(is_numeric_data(cell) for cell in row.values[1:]):
                                data_start_idx = idx
                                break
                                
                        if data_start_idx == 0 and len(df) > 1:
                            data_start_idx = 1
                            
                        # Squish the header rows into a single row by taking the last non-empty value in each column
                        headers = []
                        for col_idx in df.columns:
                            col_header = ""
                            for r_idx in range(data_start_idx):
                                cell = str(df.iloc[r_idx, col_idx]).strip().replace('\n', ' ')
                                if cell:
                                    col_header = cell
                            headers.append(col_header)
                            
                        clean_rows = []
                        # Add our squished header
                        clean_rows.append(" | ".join(headers))
                        
                        # Add the remaining data rows
                        for idx in range(data_start_idx, len(df)):
                            row = df.iloc[idx]
                            clean_cells = [str(cell).replace('\n', ' ').strip() if cell is not None else "" for cell in row]
                            
                            # strip internal whitespace between $ and numbers
                            for c_idx in range(len(clean_cells)):
                                clean_cells[c_idx] = re.sub(r'\$\s+', '$', clean_cells[c_idx])
                                
                            # merge isolated $ into the next non-empty column
                            for c_idx in range(len(clean_cells) - 1):
                                if clean_cells[c_idx] == '$':
                                    # find next non-empty column to merge into
                                    for next_idx in range(c_idx + 1, len(clean_cells)):
                                        if clean_cells[next_idx]:
                                            clean_cells[next_idx] = '$' + clean_cells[next_idx]
                                            clean_cells[c_idx] = ''
                                            break
                            
                            clean_rows.append(" | ".join(clean_cells))
                            
                        header_str = "\n".join(clean_rows)
                        table_str = f"--- Page {i+1} Table {t_idx+1} ---\n{header_str}\n"
                        data["tables"].append(table_str)
                
                # Extract text only from non-table regions
                prose_page = page.filter(not_in_table)
                page_text = prose_page.extract_text(x_tolerance=3)
                
                if page_text:
                    data["text"].append(f"--- Page {i+1} (Text) ---\n{page_text}")
                    data["metadata"]["num_text_blocks"] += 1
                    
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        
    return data

