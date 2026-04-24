"""Session- and turn-level retrieval helpers."""

from typing import Any, Dict, List, Tuple

import numpy as np

from .models import TurnNode
from .utils import cosine_similarity, parse_turn_text


def chunk_conversations_session(item: Dict[str, Any], include_assistant: bool = True) -> List[Dict[str, Any]]:
    chunks = []
    haystack_sessions = item.get("haystack_sessions", [])
    haystack_dates = item.get("haystack_dates", [])

    for session_idx, session in enumerate(haystack_sessions):
        session_date = haystack_dates[session_idx] if session_idx < len(haystack_dates) else ""
        turn_texts = []

        for turn in session:
            role = turn.get("role")
            content = turn.get("content", "")
            if include_assistant:
                turn_texts.append(f"{role}: {content}")
            elif role == "user":
                turn_texts.append(content)

        if not turn_texts:
            continue

        if include_assistant:
            base_text = "\n".join(turn_texts)
            timestamp_prefix = f"[Timestamp: {session_date}]\n" if session_date else ""
            chunk_text = f"{timestamp_prefix}{base_text}" if timestamp_prefix else base_text
        else:
            combined_content = "\n".join(turn_texts)
            chunk_text = f"[Timestamp: {session_date}] user: {combined_content}"

        chunks.append(
            {
                "text": chunk_text,
                "date": session_date,
                "role": "user_assistant" if include_assistant else "user",
                "session_idx": session_idx,
            }
        )

    return chunks


def retrieve_top_sessions(
    question_embedding: np.ndarray,
    chunks: List[Dict[str, Any]],
    chunk_embeddings: List[np.ndarray],
    top_k: int = 10,
) -> List[Tuple[Dict[str, Any], float]]:
    similarities = []
    for idx, chunk_emb in enumerate(chunk_embeddings):
        similarities.append((chunks[idx], cosine_similarity(question_embedding, chunk_emb)))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


async def retrieve_top_turns_from_sessions(
    retrieved_sessions: List[Tuple[Dict[str, Any], float]],
    question_embedding: np.ndarray,
    embedding_client,
    top_k_turns: int = 15,
) -> List[TurnNode]:
    texts_to_embed = []
    turn_metadata = []

    for chunk, _session_score in retrieved_sessions:
        session_text = chunk["text"]
        session_date = chunk["date"]
        session_idx = chunk["session_idx"]

        for turn_idx, turn in enumerate(parse_turn_text(session_text)):
            turn_text = f"[Timestamp: {session_date}]\n{turn['role']}: {turn['content']}"
            texts_to_embed.append(turn_text)
            turn_metadata.append(
                {
                    "text": turn_text,
                    "date": session_date,
                    "session_idx": session_idx,
                    "turn_idx": turn_idx,
                    "role": turn["role"],
                }
            )

    if not texts_to_embed:
        return []

    turn_embeddings = await embedding_client.get_embeddings(texts_to_embed, batch_size=200)

    turn_nodes = []
    for metadata, embedding in zip(turn_metadata, turn_embeddings):
        embedding_np = np.array(embedding) if isinstance(embedding, list) else embedding
        similarity = cosine_similarity(question_embedding, embedding_np)
        turn_id = f"session_{metadata['session_idx']}_turn_{metadata['turn_idx']}"
        turn_nodes.append(
            TurnNode(
                turn_id=turn_id,
                text=metadata["text"],
                embedding=embedding_np,
                date=metadata["date"],
                session_idx=metadata["session_idx"],
                turn_idx=metadata["turn_idx"],
                role=metadata["role"],
                similarity_to_query=similarity,
            )
        )

    turn_nodes.sort(key=lambda n: n.similarity_to_query, reverse=True)
    return turn_nodes[:top_k_turns]
