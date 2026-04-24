"""Shared data models for CoM."""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class TurnNode:
    turn_id: str
    text: str
    embedding: np.ndarray
    date: str
    session_idx: int
    turn_idx: int
    role: str
    similarity_to_query: float = 0.0


@dataclass
class EvidenceChain:
    anchor: TurnNode
    nodes: List[TurnNode]
    marginal_gains: List[float]


@dataclass
class CoMResult:
    evidence_chains: List[EvidenceChain]
    main_chain: List[TurnNode]
    full_chain: List[TurnNode]
    marginal_gains: List[float]
    pruned_count: int
    unassigned_nodes: List[TurnNode] = field(default_factory=list)
