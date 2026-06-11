"""
src: Modular pipeline engine for Naive RAG vs. Table-Aware RAG (FT-RAG) 
     comparative benchmarking.
"""

# Import the main orchestration pipeline
from .pipeline import run_pipeline

# Import component APIs for discrete access/testing
from .ingestion import parse_document_linear, parse_document_table_aware
from .indexing import index_naive_chunks, index_table_aware_rows
from .retrieval import retrieve_naive, retrieve_table_aware
from .evaluation import run_evaluation_suite, RowHitMetric, NAVMetric

__all__ = [
    "run_pipeline",
    "parse_document_linear",
    "parse_document_table_aware",
    "index_naive_chunks",
    "index_table_aware_rows",
    "retrieve_naive",
    "retrieve_table_aware",
    "run_evaluation_suite",
    "RowHitMetric",
    "NAVMetric",
]
