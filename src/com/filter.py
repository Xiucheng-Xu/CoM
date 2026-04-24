"""CoM filtering primitives."""

from datetime import datetime
from typing import List, Literal, Tuple

import numpy as np

from .models import CoMResult, EvidenceChain, TurnNode
from .utils import cosine_similarity


ContextMode = Literal["concatenated", "last_node", "average"]
ScoreMode = Literal["multiply", "query_only", "context_only", "weighted"]


def parse_date_for_sort(date_str: str) -> datetime:
    if not date_str:
        return datetime.min

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d %B %Y",
        "%B %d, %Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    return datetime.min


def sort_nodes_by_time(nodes: List[TurnNode]) -> List[TurnNode]:
    return sorted(nodes, key=lambda n: parse_date_for_sort(n.date))


def _concatenate_chain_text(chain: List[TurnNode]) -> str:
    return "\n\n".join(node.text for node in chain) if chain else ""


def _compute_average_embedding(chain: List[TurnNode]) -> np.ndarray:
    if not chain:
        return np.array([])
    return np.mean(np.array([node.embedding for node in chain]), axis=0)


async def _get_context_embedding_async(
    chain: List[TurnNode],
    embedding_client,
    context_mode: ContextMode,
) -> np.ndarray:
    if context_mode == "last_node":
        return chain[-1].embedding
    if context_mode == "average":
        return _compute_average_embedding(chain)

    context_text = _concatenate_chain_text(chain)
    context_embedding = await embedding_client.get_embedding(context_text)
    return np.array(context_embedding)


def _get_context_embedding_sync(
    chain: List[TurnNode],
    context_mode: ContextMode,
) -> np.ndarray:
    if context_mode == "last_node":
        return chain[-1].embedding
    return _compute_average_embedding(chain)


def compute_gating_score(
    candidate: TurnNode,
    context_embedding: np.ndarray,
    score_mode: ScoreMode = "multiply",
    score_alpha: float = 0.5,
) -> float:
    sim_to_query = candidate.similarity_to_query
    sim_to_context = cosine_similarity(candidate.embedding, context_embedding)

    if score_mode == "query_only":
        return sim_to_query
    if score_mode == "context_only":
        return sim_to_context
    if score_mode == "weighted":
        return score_alpha * sim_to_query + (1 - score_alpha) * sim_to_context
    return sim_to_query * sim_to_context


def sample_anchors(
    candidate_nodes: List[TurnNode],
    num_anchors: int,
    sampling_method: str = "top_k",
) -> List[TurnNode]:
    if not candidate_nodes:
        return []

    num_anchors = min(num_anchors, len(candidate_nodes))

    if sampling_method == "top_k":
        sorted_nodes = sorted(candidate_nodes, key=lambda n: n.similarity_to_query, reverse=True)
        return sorted_nodes[:num_anchors]

    if sampling_method == "weighted":
        similarities = np.array([node.similarity_to_query for node in candidate_nodes])
        similarities = np.maximum(similarities, 0)
        probs = (
            np.ones(len(candidate_nodes)) / len(candidate_nodes)
            if similarities.sum() == 0
            else similarities / similarities.sum()
        )
        indices = np.random.choice(len(candidate_nodes), size=num_anchors, replace=False, p=probs)
        return [candidate_nodes[i] for i in indices]

    raise ValueError(f"Unknown sampling method: {sampling_method}")


async def build_single_chain_async(
    anchor: TurnNode,
    candidate_nodes: List[TurnNode],
    embedding_client,
    blocking_ratio: float,
    context_mode: ContextMode = "concatenated",
    score_mode: ScoreMode = "multiply",
    score_alpha: float = 0.5,
) -> Tuple[List[TurnNode], List[float]]:
    chain = [anchor]
    marginal_gains = [anchor.similarity_to_query]
    remaining = [node for node in candidate_nodes if node.turn_id != anchor.turn_id]
    prev_score = anchor.similarity_to_query

    while remaining:
        context_embedding = await _get_context_embedding_async(chain, embedding_client, context_mode)

        scores = []
        for node in remaining:
            score = compute_gating_score(
                candidate=node,
                context_embedding=context_embedding,
                score_mode=score_mode,
                score_alpha=score_alpha,
            )
            scores.append((node, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_node, best_score = scores[0]
        if best_score < prev_score * blocking_ratio:
            break

        chain.append(best_node)
        marginal_gains.append(best_score)
        remaining.remove(best_node)
        prev_score = best_score

    return chain, marginal_gains


def build_single_chain(
    anchor: TurnNode,
    candidate_nodes: List[TurnNode],
    blocking_ratio: float,
    context_mode: ContextMode = "last_node",
    score_mode: ScoreMode = "multiply",
    score_alpha: float = 0.5,
) -> Tuple[List[TurnNode], List[float]]:
    chain = [anchor]
    marginal_gains = [anchor.similarity_to_query]
    remaining = [node for node in candidate_nodes if node.turn_id != anchor.turn_id]
    prev_score = anchor.similarity_to_query

    while remaining:
        context_embedding = _get_context_embedding_sync(chain, context_mode)

        scores = []
        for node in remaining:
            score = compute_gating_score(
                candidate=node,
                context_embedding=context_embedding,
                score_mode=score_mode,
                score_alpha=score_alpha,
            )
            scores.append((node, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_node, best_score = scores[0]
        if best_score < prev_score * blocking_ratio:
            break

        chain.append(best_node)
        marginal_gains.append(best_score)
        remaining.remove(best_node)
        prev_score = best_score

    return chain, marginal_gains


async def com_filter_async(
    candidate_nodes: List[TurnNode],
    query_embedding: np.ndarray,
    blocking_ratio: float = 0.5,
    num_anchors: int = 3,
    sampling_method: str = "top_k",
    max_chain_length: int = 20,
    context_mode: ContextMode = "concatenated",
    embedding_client=None,
    score_mode: ScoreMode = "multiply",
    score_alpha: float = 0.5,
) -> CoMResult:
    _ = query_embedding, max_chain_length

    if not candidate_nodes:
        return CoMResult(
            evidence_chains=[],
            main_chain=[],
            full_chain=[],
            marginal_gains=[],
            pruned_count=0,
            unassigned_nodes=[],
        )

    anchors = sample_anchors(
        candidate_nodes=candidate_nodes,
        num_anchors=num_anchors,
        sampling_method=sampling_method,
    )

    evidence_chains: List[EvidenceChain] = []
    for anchor in anchors:
        chain, gains = await build_single_chain_async(
            anchor=anchor,
            candidate_nodes=candidate_nodes,
            embedding_client=embedding_client,
            blocking_ratio=blocking_ratio,
            context_mode=context_mode,
            score_mode=score_mode,
            score_alpha=score_alpha,
        )
        evidence_chains.append(EvidenceChain(anchor=anchor, nodes=chain, marginal_gains=gains))

    evidence_chains.sort(key=lambda x: x.anchor.similarity_to_query, reverse=True)

    merged_chain: List[TurnNode] = []
    merged_gains: List[float] = []
    for chain in evidence_chains:
        merged_chain.extend(chain.nodes)
        merged_gains.extend(chain.marginal_gains)

    pruned_count = len(candidate_nodes) * len(anchors) - len(merged_chain)

    return CoMResult(
        evidence_chains=evidence_chains,
        main_chain=merged_chain,
        full_chain=merged_chain,
        marginal_gains=merged_gains,
        pruned_count=pruned_count,
        unassigned_nodes=[],
    )


def com_filter(
    candidate_nodes: List[TurnNode],
    query_embedding: np.ndarray,
    blocking_ratio: float = 0.5,
    num_anchors: int = 3,
    sampling_method: str = "top_k",
    max_chain_length: int = 20,
    context_mode: ContextMode = "last_node",
    embedding_client=None,
    score_mode: ScoreMode = "multiply",
    score_alpha: float = 0.5,
) -> CoMResult:
    _ = query_embedding, max_chain_length, embedding_client

    if not candidate_nodes:
        return CoMResult(
            evidence_chains=[],
            main_chain=[],
            full_chain=[],
            marginal_gains=[],
            pruned_count=0,
            unassigned_nodes=[],
        )

    anchors = sample_anchors(
        candidate_nodes=candidate_nodes,
        num_anchors=num_anchors,
        sampling_method=sampling_method,
    )

    evidence_chains: List[EvidenceChain] = []
    for anchor in anchors:
        chain, gains = build_single_chain(
            anchor=anchor,
            candidate_nodes=candidate_nodes,
            blocking_ratio=blocking_ratio,
            context_mode=context_mode,
            score_mode=score_mode,
            score_alpha=score_alpha,
        )
        evidence_chains.append(EvidenceChain(anchor=anchor, nodes=chain, marginal_gains=gains))

    evidence_chains.sort(key=lambda x: x.anchor.similarity_to_query, reverse=True)

    merged_chain: List[TurnNode] = []
    merged_gains: List[float] = []
    for chain in evidence_chains:
        merged_chain.extend(chain.nodes)
        merged_gains.extend(chain.marginal_gains)

    pruned_count = len(candidate_nodes) * len(anchors) - len(merged_chain)

    return CoMResult(
        evidence_chains=evidence_chains,
        main_chain=merged_chain,
        full_chain=merged_chain,
        marginal_gains=merged_gains,
        pruned_count=pruned_count,
        unassigned_nodes=[],
    )


def format_com_context(com_result: CoMResult, include_unassigned: bool = True) -> str:
    _ = include_unassigned

    if not com_result.evidence_chains:
        return ""

    formatted_parts = []
    for i, chain in enumerate(com_result.evidence_chains, 1):
        chain_text = "\n\n".join(node.text for node in chain.nodes)
        formatted_parts.append(f"=== Evidence Chain {i} ===\n{chain_text}")

    return "\n\n".join(formatted_parts)
