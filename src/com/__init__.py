"""
CoM module - shared implementation for the LME and LoComo runners.
"""

from .models import TurnNode, CoMResult, EvidenceChain
from .utils import cosine_similarity, load_dataset, remove_think_tags, parse_turn_text
from .retrieval import chunk_conversations_session, retrieve_top_sessions, retrieve_top_turns_from_sessions
from .filter import (
    compute_gating_score,
    com_filter,
    com_filter_async,
    format_com_context,
    sample_anchors,
    build_single_chain,
    build_single_chain_async,
    ContextMode,
    ScoreMode,
)
from .pipeline import process_sample, run_pipeline

__all__ = [
    "TurnNode",
    "CoMResult",
    "EvidenceChain",
    "cosine_similarity",
    "load_dataset",
    "remove_think_tags",
    "parse_turn_text",
    "chunk_conversations_session",
    "retrieve_top_sessions",
    "retrieve_top_turns_from_sessions",
    "compute_gating_score",
    "com_filter",
    "com_filter_async",
    "format_com_context",
    "sample_anchors",
    "build_single_chain",
    "build_single_chain_async",
    "ContextMode",
    "ScoreMode",
    "process_sample",
    "run_pipeline",
]
