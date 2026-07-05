"""
This module implements the evaluation metrics for the RAG pipeline.

Classes:
    RowHitMetric: Computes Recall@K for table rows.
    NAVMetric: Computes Numerical Atomic Verification (NAV) Metric.
Functions:
    run_evaluation_suite: Runs the evaluation suite over the pipeline results.
"""
import re

class RowHitMetric:
    """
    Computes Recall@K for table rows.
    Checks if the expected table ID or row was present in the retrieved chunks.
    """
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def score(self, expected_table_id: str, retrieved_chunks: list) -> int:
        """
        Returns 1 if the expected_table_id is found in the metadata of any
        of the retrieved_chunks, else 0.

        Args:
            expected_table_id: The expected table ID.
            retrieved_chunks: The retrieved chunks.
        
        Returns:
            1 if the expected table ID is found, 0 otherwise.
        """
        if not expected_table_id:
            return 0
            
        for chunk in retrieved_chunks[:self.top_k]:
            metadata = chunk.get("metadata", {})
            table_id = metadata.get("table_id")
            if table_id and expected_table_id in table_id:
                return 1
        return 0

class NAVMetric:
    """
    Numerical Atomic Verification (NAV) Metric.
    Ensures that the exact numerical ground truth is present in the generated answer.
    Supports multiple acceptable formats separated by '|'.
    """
    def score(self, expected_value: str, generated_answer: str) -> int:
        """
        Returns 1 if any of the expected numerical values (separated by '|')
        is present in the generated answer.

        Args:
            expected_value: The expected numerical value.
            generated_answer: The generated answer.
        
        Returns:
            1 if the expected value is found, 0 otherwise.
        """
        if not expected_value or not generated_answer:
            return 0
            
        synonyms = [s.strip() for s in str(expected_value).split("|")]
        norm_answer = str(generated_answer).replace(",", "").replace(" ", "").lower()
        
        for syn in synonyms:
            if not syn:
                continue
            # Remove commas and spaces for robust checking
            norm_syn = syn.replace(",", "").replace(" ", "").lower()
            if norm_syn in norm_answer:
                return 1
            # Check if original expected value string is in original answer
            if syn.lower() in str(generated_answer).lower():
                return 1
                
        return 0

def run_evaluation_suite(pipeline_results: list) -> dict:
    """
    Computes metrics over the pipeline results.
    Assumes pipeline_results is a list of dicts with:
    - Expected_Table_ID
    - Expected_Answer
    - Retrieved_Chunks (list of dicts)
    - Generated_Answer (str)
    - Tier (str)
    
    Args:
        pipeline_results: The pipeline results.
    
    Returns:
        Aggregated metrics.
    """
    row_hit_evaluator = RowHitMetric()
    nav_evaluator = NAVMetric()
    
    metrics_by_tier = {}
    overall_row_hit = 0
    overall_nav = 0
    total_evaluated = 0
    
    for result in pipeline_results:
        tier = result.get("Tier", "Unknown")
        expected_table = result.get("Expected_Table_ID", "")
        expected_answer = result.get("Expected_Answer", "")
        chunks = result.get("Retrieved_Chunks", [])
        answer = result.get("Generated_Answer", "")
        
        if tier not in metrics_by_tier:
            metrics_by_tier[tier] = {"total": 0, "row_hit": 0, "nav": 0}
            
        metrics_by_tier[tier]["total"] += 1
        total_evaluated += 1
        
        # Calculate scores
        row_score = row_hit_evaluator.score(expected_table, chunks)
        nav_score = nav_evaluator.score(expected_answer, answer)
        
        metrics_by_tier[tier]["row_hit"] += row_score
        metrics_by_tier[tier]["nav"] += nav_score
        
        overall_row_hit += row_score
        overall_nav += nav_score
        
    # Calculate percentages
    summary = {
        "Overall": {
            "Total": total_evaluated,
            "RowHit_Accuracy": (overall_row_hit / total_evaluated) if total_evaluated > 0 else 0,
            "NAV_Accuracy": (overall_nav / total_evaluated) if total_evaluated > 0 else 0
        },
        "By_Tier": {}
    }
    
    for tier, data in metrics_by_tier.items():
        total = data["total"]
        summary["By_Tier"][tier] = {
            "Total": total,
            "RowHit_Accuracy": (data["row_hit"] / total) if total > 0 else 0,
            "NAV_Accuracy": (data["nav"] / total) if total > 0 else 0
        }
        
    return summary
