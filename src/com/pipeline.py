"""
CoM full pipeline for LongMemEval-style samples.
"""

import asyncio
import json
import os
import time
import traceback
from typing import Any, Dict, List

import numpy as np

from src.llm.prompt_baseline import BASELINE_PROMPT, BASELINE_SYSTEM_PROMPT

from .filter import com_filter_async, format_com_context
from .retrieval import chunk_conversations_session, retrieve_top_sessions, retrieve_top_turns_from_sessions
from .utils import remove_think_tags



async def process_sample(
    idx: int,
    item: Dict[str, Any],
    llm_client,
    embedding_client,
    top_k_turns: int,
    include_assistant: bool,
    total_count: int,
    blocking_ratio: float = 0.5,
    num_anchors: int = 3,
    context_mode: str = "concatenated",
) -> Dict[str, Any]:
    question_id = item.get("question_id", f"unknown_{idx}")
    print(f"\n[{idx + 1}/{total_count}] Processing question: {question_id}")
    start_total = time.time()

    try:
        retrieval_data = await _execute_retrieval(
            item=item,
            embedding_client=embedding_client,
            top_k_turns=top_k_turns,
            include_assistant=include_assistant,
            question_id=question_id,
        )

        start_com = time.time()
        com_result = await com_filter_async(
            candidate_nodes=retrieval_data["turn_nodes"],
            query_embedding=retrieval_data["question_embedding"],
            embedding_client=embedding_client,
            blocking_ratio=blocking_ratio,
            num_anchors=num_anchors,
            sampling_method="top_k",
            context_mode=context_mode,
            score_mode="multiply",
        )
        com_time = time.time() - start_com
        print(
            f"  [{question_id}] CoM: {com_time:.3f}s, "
            f"chains={len(com_result.evidence_chains)}, "
            f"nodes={len(com_result.main_chain)}, "
            f"pruned={com_result.pruned_count}"
        )

        context = format_com_context(com_result)
        messages = [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": BASELINE_PROMPT.format(
                    context=context,
                    question=retrieval_data["question_with_date"],
                ),
            },
        ]

        start_llm = time.time()
        hypothesis, usage = await llm_client.generate(messages=messages, return_usage=True)
        hypothesis = remove_think_tags(hypothesis)
        llm_time = time.time() - start_llm
        total_time = time.time() - start_total

        print(f"  [{question_id}] LLM: {llm_time:.2f}s, total={total_time:.2f}s")

        return {
            "result": {
                "question_id": question_id,
                "hypothesis": hypothesis,
                "retrieved_text": context,
            },
            "stats": {
                "tokens": usage.total_tokens if usage else 0,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "time": total_time,
                "com_time": com_time,
                "llm_time": llm_time,
                "main_chain_size": len(com_result.main_chain),
                "full_chain_size": len(com_result.full_chain),
                "pruned_count": com_result.pruned_count,
                "success": True,
            },
        }

    except Exception as e:
        print(f"  [{question_id}] Error: {e}")
        traceback.print_exc()
        return _create_error_result(question_id, str(e), time.time() - start_total)


def _create_error_result(question_id: str, error: str, elapsed_time: float = 0) -> Dict[str, Any]:
    return {
        "result": {
            "question_id": question_id,
            "hypothesis": "",
            "error": error,
            "retrieved_text": "",
        },
        "stats": {"tokens": 0, "time": elapsed_time, "success": False},
    }


async def _execute_retrieval(
    item: Dict[str, Any],
    embedding_client,
    top_k_turns: int,
    include_assistant: bool,
    question_id: str,
) -> Dict[str, Any]:
    question = item.get("question", "")
    question_date = item.get("question_date", "")
    question_with_date = f"[Timestamp: {question_date}] {question}" if question_date else question

    chunks = chunk_conversations_session(item, include_assistant=include_assistant)
    chunk_texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings, question_embedding = await asyncio.gather(
        embedding_client.get_embeddings(chunk_texts, batch_size=200),
        embedding_client.get_embedding(question_with_date),
    )
    question_embedding_np = np.array(question_embedding)

    retrieved_sessions = retrieve_top_sessions(
        question_embedding=question_embedding_np,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        top_k=min(10, len(chunks)),
    )

    turn_nodes = await retrieve_top_turns_from_sessions(
        retrieved_sessions=retrieved_sessions,
        question_embedding=question_embedding_np,
        embedding_client=embedding_client,
        top_k_turns=top_k_turns,
    )
    print(f"  [{question_id}] Turn retrieval: {len(turn_nodes)} turns")

    return {
        "question": question,
        "question_date": question_date,
        "question_with_date": question_with_date,
        "question_embedding": question_embedding_np,
        "retrieved_sessions": retrieved_sessions,
        "turn_nodes": turn_nodes,
    }


async def run_pipeline(
    data: List[Dict[str, Any]],
    llm_client,
    embedding_client,
    output_file: str,
    stats_file: str,
    top_k_turns: int = 15,
    include_assistant: bool = True,
    max_concurrent_samples: int = 20,
    blocking_ratio: float = 0.5,
    num_anchors: int = 3,
    context_mode: str = "concatenated",
) -> None:
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(stats_file) or ".", exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Starting CoM full pipeline")
    print(f"Samples: {len(data)}")
    print(
        "CoM: "
        f"num_anchors={num_anchors}, "
        f"blocking_ratio={blocking_ratio}, "
        "sampling=top_k, "
        "score_mode=multiply"
    )
    print(f"{'=' * 60}\n")

    start_time = time.time()
    semaphore = asyncio.Semaphore(max_concurrent_samples)

    async def process_with_semaphore(idx, sample):
        async with semaphore:
            return await process_sample(
                idx=idx,
                item=sample,
                llm_client=llm_client,
                embedding_client=embedding_client,
                top_k_turns=top_k_turns,
                include_assistant=include_assistant,
                total_count=len(data),
                blocking_ratio=blocking_ratio,
                num_anchors=num_anchors,
                context_mode=context_mode,
            )

    all_results = await asyncio.gather(*(process_with_semaphore(idx, sample) for idx, sample in enumerate(data)))

    results = [r["result"] for r in all_results]
    success_stats = [r["stats"] for r in all_results if r["stats"].get("success")]

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    elapsed_time = time.time() - start_time
    n_success = len(success_stats)

    total_tokens = sum(s.get("tokens", 0) for s in success_stats)
    prompt_tokens = sum(s.get("prompt_tokens", 0) for s in success_stats)
    completion_tokens = sum(s.get("completion_tokens", 0) for s in success_stats)

    com_params = {
        "num_anchors": num_anchors,
        "blocking_ratio": blocking_ratio,
        "sampling_method": "top_k",
        "context_mode": context_mode,
        "score_mode": "multiply",
    }
    stats = {
        "status": "completed",
        "method": "com_full",
        "top_k_turns": top_k_turns,
        "com_params": com_params,
        "total_samples": len(results),
        "successful_samples": n_success,
        "total_tokens": round(total_tokens / 1000, 2),
        "prompt_tokens": round(prompt_tokens / 1000, 2),
        "completion_tokens": round(completion_tokens / 1000, 2),
        "wall_clock_time_seconds": round(elapsed_time, 2),
        "avg_tokens_per_sample": round(total_tokens / n_success / 1000, 2) if n_success else 0,
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"CoM evaluation completed: {output_file}")
    print(f"Success: {n_success}/{len(results)}")
    print(f"Total tokens: {stats['total_tokens']}k, wall time: {elapsed_time:.2f}s")
    print(f"{'=' * 60}\n")
