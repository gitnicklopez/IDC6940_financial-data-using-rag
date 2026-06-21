'''
This script generates a CSV file containing statistics about the ingested documents.

Functions:
    - main(): Main function to generate corpus statistics.
        - Inputs: None
        - Outputs: CSV file containing statistics about the ingested documents.
'''
import os
import sys
import csv

# Add the project root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion import parse_document_table_aware
from src.indexing import index_table_aware_rows

def main():
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "corpus_statistics.csv")
    
    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]
    # Sort files so FY goes in order
    pdf_files.sort()
    
    results = []
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        pdf_path = os.path.join(docs_dir, pdf_file)
        
        # Parse Document
        parsed_data = parse_document_table_aware(pdf_path)
        
        # Index to get chunk counts
        indexed_data = index_table_aware_rows(parsed_data)
        
        # Extract Fiscal Year from filename (e.g., 'UWF-Financial-Statement-FY-20-21.pdf' -> 'FY 20-21')
        fy = "Unknown"
        if "FY-" in pdf_file:
            parts = pdf_file.split("FY-")
            if len(parts) > 1:
                fy = "FY " + parts[1].replace(".pdf", "")
        
        metadata = parsed_data.get("metadata", {})
        
        stat = {
            "Filename": pdf_file,
            "Fiscal_Year": fy,
            "Total_Pages": metadata.get("num_pages", 0),
            "Total_Text_Blocks": metadata.get("num_text_blocks", 0),
            "Total_Tables": metadata.get("num_tables", 0),
            "Total_Table_Rows": metadata.get("num_rows", 0),
            "Narrative_Chunks": len(indexed_data.get("text", [])),
            "SRSE_Chunks": len(indexed_data.get("tables", []))
        }
        results.append(stat)
        
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Statistics successfully written to {csv_path}")

if __name__ == "__main__":
    main()
