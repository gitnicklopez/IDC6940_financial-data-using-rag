"""
This module implements the evaluation metrics and orchestration for the RAG pipeline.

It compares a Naive RAG approach versus a Table-Aware RAG approach.
Outputs results to eval_results.csv.
"""
import os
import sys
import csv
import time

# Add the project root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_pipeline
from src.retrieval import retrieve_table_aware, retrieve_naive
from src.generation import generate_response

class RowHitMetric:
    """
    Computes Recall for tables and pages.
    Checks if the expected document and location (page/table) were present in the retrieved chunks.
    """
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def score(self, expected_corpus_file: str, expected_location: str, retrieved_chunks: list) -> int:
        if not expected_corpus_file or not expected_location:
            return 0
            
        import re
        expected_locations = [loc.strip() for loc in expected_location.split(",")]
        expected_files = [f.strip() for f in expected_corpus_file.split(",")]
        
        expected_pages = []
        for loc in expected_locations:
            match = re.search(r'\d+', loc)
            if match:
                expected_pages.append(match.group())
            
        for chunk in retrieved_chunks[:self.top_k]:
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "")
            if filename not in expected_files:
                continue
                
            page = str(metadata.get("page", ""))
            table_id = metadata.get("table_id", "")
            
            if page and page in expected_pages:
                return 1
            
            for loc in expected_locations:
                if (table_id and loc.lower() in table_id.lower()) or (page and loc.lower() in page.lower()):
                    return 1
        return 0

class NAVMetric:
    """
    Numerical Atomic Verification (NAV) Metric.
    Ensures that the exact numerical ground truth is present in the generated answer.
    Supports multiple acceptable formats separated by '|'.
    """
    def score(self, expected_value: str, generated_answer: str) -> int:
        if not expected_value or not generated_answer:
            return 0
            
        # 1. Try numerical value verification
        try:
            expected_candidates = []
            synonyms = [s.strip() for s in str(expected_value).split("|")]
            for syn in synonyms:
                expected_candidates.extend(self._extract_numerical_candidates(syn))
                
            generated_candidates = self._extract_numerical_candidates(generated_answer)
            
            if expected_candidates and generated_candidates:
                for exp_val in expected_candidates:
                    exp_is_year = 1990.0 <= exp_val <= 2040.0
                    for gen_val in generated_candidates:
                        # If expected value is not a year, don't match it with a year in the generated answer
                        if not exp_is_year and (1990.0 <= gen_val <= 2040.0):
                            continue
                        
                        if exp_val == 0:
                            diff = abs(gen_val)
                        else:
                            diff = abs(exp_val - gen_val) / exp_val
                            
                        if diff <= 0.01:  # 1% tolerance
                            return 1
        except Exception:
            # Fallback if parsing fails
            pass

        # 2. Fallback: Exact string matching
        synonyms = [s.strip() for s in str(expected_value).split("|")]
        norm_answer = str(generated_answer).replace(",", "").replace(" ", "").lower()
        for syn in synonyms:
            if not syn:
                continue
            norm_syn = syn.replace(",", "").replace(" ", "").lower()
            if norm_syn in norm_answer:
                return 1
            if syn.lower() in str(generated_answer).lower():
                return 1
                
        return 0

    def _extract_numerical_candidates(self, text: str) -> list:
        import re
        if not text:
            return []
        
        text = text.lower()
        text = text.replace('–', '-').replace('—', '-')
        
        # Find all scale words present anywhere in the text
        scale_words = re.findall(r'\b(thousand|thousands|million|millions|billion|billions|k|m|b)\b', text)
        global_multipliers = set()
        for w in scale_words:
            w = w.strip()
            if "billion" in w or w == "b":
                global_multipliers.add(1_000_000_000.0)
            elif "million" in w or w == "m":
                global_multipliers.add(1_000_000.0)
            elif "thousand" in w or w == "k":
                global_multipliers.add(1_000.0)
        global_multipliers.add(1.0)  # Always include unscaled
        
        # Find all numbers, potentially with immediate scale words
        pattern = r'(-?\d[\d,]*\.?\d*)\s*(thousand|thousands|million|millions|billion|billions|k|m|b)?\b'
        matches = re.findall(pattern, text)
        
        candidates = set()
        for num_str, local_scale in matches:
            if num_str.endswith('.'):
                num_str = num_str[:-1]
            if not num_str or num_str == '-':
                continue
                
            try:
                clean_num_str = num_str.replace(",", "")
                base_val = float(clean_num_str)
            except ValueError:
                continue
                
            if local_scale:
                local_scale = local_scale.strip()
                mult = 1.0
                if "billion" in local_scale or local_scale == "b":
                    mult = 1_000_000_000.0
                elif "million" in local_scale or local_scale == "m":
                    mult = 1_000_000.0
                elif "thousand" in local_scale or local_scale == "k":
                    mult = 1_000.0
                val = base_val * mult
                candidates.add(abs(val))
            else:
                for mult in global_multipliers:
                    candidates.add(abs(base_val * mult))
                    
        return list(candidates)

def format_context_for_md(retrieved_chunks):
    md_context_parts = []
    for c_idx, chunk in enumerate(retrieved_chunks):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        table_id = metadata.get("table_id")
        row_index = metadata.get("row_index")
        filename_meta = metadata.get("filename", "unknown")
        
        meta_str = f"Source: `{source}`, File: `{filename_meta}`, Page: `{page}`"
        if table_id:
            meta_str += f", Table ID: `{table_id}`"
        if row_index:
            meta_str += f", Row Index: `{row_index}`"
            
        chunk_text = chunk.get('text', '')
        if source == "reconstructed_table" or ("|" in chunk_text and "-|-" in chunk_text) or ("|" in chunk_text and "---" in chunk_text):
            chunk_md = f"### Chunk {c_idx + 1} ({meta_str})\n\n{chunk_text}"
        else:
            chunk_md = f"### Chunk {c_idx + 1} ({meta_str})\n```\n{chunk_text}\n```"
        md_context_parts.append(chunk_md)
    return "\n\n".join(md_context_parts)

def main():
    print("Starting RAG evaluation generation...")
    
    # Set up directory paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf_dir = os.path.join(base_dir, "data", "corpus")
    eval_dir = os.path.join(base_dir, "data", "eval")
    responses_dir = os.path.join(eval_dir, "responses")
    questions_csv = os.path.join(eval_dir, "questions.csv")
    output_csv = os.path.join(eval_dir, "eval_results.csv")
    
    # Ensure directories exist
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(responses_dir, exist_ok=True)
    
    if not os.path.exists(questions_csv):
        print(f"Error: Questions file not found at {questions_csv}")
        return
        
    print(f"Loading and indexing PDFs from {pdf_dir}...")
    try:
        indices = run_pipeline(pdf_dir)
        naive_index = indices.get("naive_index", [])
        text_index = indices.get("text_index", [])
        table_row_index = indices.get("table_row_index", [])
        print(f"Pipeline finished. Loaded {len(naive_index)} naive chunks, {len(text_index)} text chunks, and {len(table_row_index)} table chunks.")
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return

    # Read questions
    questions = []
    with open(questions_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)
            
    print(f"Loaded {len(questions)} questions for evaluation.")
    
    row_hit_evaluator = RowHitMetric()
    nav_evaluator = NAVMetric()
    
    results = []
    
    for idx, q_row in enumerate(questions):
        q_id = q_row.get("Q_ID", "")
        tier = q_row.get("Tier", "")
        question_text = q_row.get("Question", "")
        expected_corpus_file = q_row.get("corpus_file", "")
        expected_location = q_row.get("Expected_Table_ID", "")
        expected_answer = q_row.get("Expected_Answer", "")
        
        print(f"Processing ({idx+1}/{len(questions)}): [{q_id}] {question_text}")
        
        # --- NAIVE RAG ---
        naive_chunks = retrieve_naive(
            query=question_text,
            indexed_chunks=naive_index,
            top_k=5
        )
        naive_context = "\n\n".join([chunk.get("text", "") for chunk in naive_chunks])
        
        try:
            naive_answer = generate_response(prompt=question_text, context=naive_context)
        except Exception as e:
            print(f"Error generating naive response for {q_id}: {e}")
            naive_answer = f"ERROR: {e}"
            
        naive_row_hit = row_hit_evaluator.score(expected_corpus_file, expected_location, naive_chunks)
        naive_nav = nav_evaluator.score(expected_answer, naive_answer)
        
        # --- TABLE-AWARE RAG ---
        ta_chunks = retrieve_table_aware(
            query=question_text,
            text_index=text_index,
            table_row_index=table_row_index,
            top_k=5
        )
        ta_context = "\n\n".join([chunk.get("text", "") for chunk in ta_chunks])
        
        try:
            ta_answer = generate_response(prompt=question_text, context=ta_context)
        except Exception as e:
            print(f"Error generating table-aware response for {q_id}: {e}")
            ta_answer = f"ERROR: {e}"
            
        ta_row_hit = row_hit_evaluator.score(expected_corpus_file, expected_location, ta_chunks)
        ta_nav = nav_evaluator.score(expected_answer, ta_answer)
        
        # Save to Markdown
        safe_q_id = q_id if q_id else f"Q_Unknown_{idx+1}"
        md_filename = f"{safe_q_id}.md"
        md_path = os.path.join(responses_dir, md_filename)
        
        md_content = f"# Question ID: {q_id}\n**Tier:** {tier}\n\n## Question\n{question_text}\n\n"
        md_content += f"## Ground Truth\n- **Expected File:** `{expected_corpus_file}`\n- **Expected Location:** `{expected_location}`\n- **Expected Answer:** `{expected_answer}`\n\n"
        md_content += f"## Table-Aware RAG (RowHit: {ta_row_hit}, NAV: {ta_nav})\n"
        md_content += f"### Generated Answer\n{ta_answer}\n\n### Retrieved Context\n{format_context_for_md(ta_chunks)}\n\n"
        md_content += f"---\n\n"
        md_content += f"## Naive RAG (RowHit: {naive_row_hit}, NAV: {naive_nav})\n"
        md_content += f"### Generated Answer\n{naive_answer}\n\n### Retrieved Context\n{format_context_for_md(naive_chunks)}\n"
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        results.append({
            "Q_ID": q_id,
            "Tier": tier,
            "Question": question_text,
            "Expected_Corpus": expected_corpus_file,
            "Expected_Location": expected_location,
            "Expected_Answer": expected_answer,
            "Naive_Answer": naive_answer,
            "Naive_RowHit": naive_row_hit,
            "Naive_NAV": naive_nav,
            "TableAware_Answer": ta_answer,
            "TableAware_RowHit": ta_row_hit,
            "TableAware_NAV": ta_nav,
            "Response_File": md_filename
        })
        
    # Save results
    print(f"Saving generated results to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["Q_ID", "Tier", "Question", "Expected_Corpus", "Expected_Location", "Expected_Answer", 
                      "Naive_Answer", "Naive_RowHit", "Naive_NAV", 
                      "TableAware_Answer", "TableAware_RowHit", "TableAware_NAV", "Response_File"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print("\n=== Evaluation Summary ===")
    total = len(results)
    if total > 0:
        naive_rh_acc = sum(r["Naive_RowHit"] for r in results) / total
        naive_nav_acc = sum(r["Naive_NAV"] for r in results) / total
        ta_rh_acc = sum(r["TableAware_RowHit"] for r in results) / total
        ta_nav_acc = sum(r["TableAware_NAV"] for r in results) / total
        print(f"Total Questions: {total}")
        print(f"Naive RAG:        RowHit Accuracy = {naive_rh_acc:.1%}, NAV Accuracy = {naive_nav_acc:.1%}")
        print(f"Table-Aware RAG:  RowHit Accuracy = {ta_rh_acc:.1%}, NAV Accuracy = {ta_nav_acc:.1%}")
    print("Evaluation generation complete!")

if __name__ == "__main__":
    main()
