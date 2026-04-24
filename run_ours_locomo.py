import asyncio
import os
import json
import argparse
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import yaml

from src.llm.async_llm_client import AsyncOpenAILLM
from src.llm.async_embedding_client import AsyncOpenAIEmbedding
from src.llm.prompt_baseline import SYSTEM_PROMPT_LOCOMO, USER_PROMPT_LOCOMO
from src.com import com_filter, com_filter_async
from src.com.utils import remove_think_tags
from src.com.models import TurnNode

import re

def _split_text(text: str) -> List[str]:
    """Split text into short retrieval units."""
    parts = re.split(r'[.!?。！？]+', text)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
    return parts if parts else [text]

def load_locomo_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def chunk_locomo_conversations_with_speaker(
    conversation: Dict[str, Any],
    mode: str = "turn"
) -> Tuple[List[Dict[str, str]], str, str]:
    """Build retrieval chunks and keep speaker labels."""
    chunks = []
    speaker_a = conversation.get('speaker_a', 'Speaker_A')
    speaker_b = conversation.get('speaker_b', 'Speaker_B')
    
    session_keys = [k for k in conversation.keys() 
                   if k.startswith('session_') and not k.endswith('_date_time')]
    
    for session_key in sorted(session_keys):
        date_time_key = f"{session_key}_date_time"
        date_time = conversation.get(date_time_key, '')
        turns = conversation.get(session_key, [])
        
        if mode == "turn":
            for turn_idx, turn in enumerate(turns):
                speaker = turn.get('speaker', '')
                text = turn.get('text', '')
                dia_id = turn.get('dia_id', '')
                
                if 'blip_caption' in turn and turn['blip_caption']:
                    text = f"{text} (image description: {turn['blip_caption']})"
                
                chunk_text = f"{speaker}: {text}"
                
                chunks.append({
                    "text": chunk_text,
                    "raw_text": text,
                    "date": date_time,
                    "session_key": session_key,
                    "dia_id": dia_id,
                    "speaker": speaker
                })

        elif mode == "session":
            turn_texts = []
            for turn in turns:
                speaker = turn.get('speaker', '')
                text = turn.get('text', '')
                if 'blip_caption' in turn and turn['blip_caption']:
                    text = f"{text} (image description: {turn['blip_caption']})"
                turn_texts.append(f"{speaker}: {text}")
            
            if turn_texts:
                combined_text = "\n".join(turn_texts)
                chunks.append({
                    "text": combined_text,
                    "date": date_time,
                    "session_key": session_key,
                    "speaker": "mixed"
                })
    
    return chunks, speaker_a, speaker_b


def convert_locomo_to_items_with_speaker(samples: List[Dict], mode: str = "turn") -> List[Dict]:
    """Expand LoCoMo samples into question-level items."""
    items = []
    allow_categories = [1, 2, 3, 4]
    
    for sample_idx, sample in enumerate(samples, 1):
        conversation = sample.get('conversation', {})
        chunks, speaker_a, speaker_b = chunk_locomo_conversations_with_speaker(conversation, mode=mode)
        
        if not chunks:
            continue
        
        qa_list = sample.get('qa', [])
        qa_counter = 0
        
        for qa in qa_list:
            category = qa.get('category')
            if category not in allow_categories:
                continue
            
            qa_counter += 1
            items.append({
                "sample_idx": sample_idx,
                "question_id": f"sample_{sample_idx}_qa_{qa_counter}",
                "question": qa.get('question', ''),
                "answer": qa.get('answer', ''),
                "category": category,
                "chunks": chunks,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b
            })
    
    return items

def detect_target_speaker(question: str, speaker_a: str, speaker_b: str) -> str:
    """Route a question to one speaker when the name is explicit."""
    question_lower = question.lower()
    speaker_a_lower = speaker_a.lower()
    speaker_b_lower = speaker_b.lower()
    
    mentions_a = speaker_a_lower in question_lower
    mentions_b = speaker_b_lower in question_lower
    
    if mentions_a and not mentions_b:
        return "speaker_a"
    elif mentions_b and not mentions_a:
        return "speaker_b"
    else:
        return "both"


def retrieve_relevant_chunks_by_speaker(
    question_embedding: np.ndarray,
    chunks: List[Dict],
    chunk_embeddings: List[np.ndarray],
    speaker_a: str,
    speaker_b: str,
    top_k: int = 25,
    target_speaker: str = "both"
) -> Tuple[List[Tuple[Dict, float]], List[Tuple[Dict, float]]]:
    """Retrieve top-k chunks for the active speaker groups."""
    speaker_a_chunks = []
    speaker_b_chunks = []
    
    for idx, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings)):
        speaker = chunk.get("speaker", "")
        if speaker == speaker_a:
            speaker_a_chunks.append((chunk, emb, idx))
        elif speaker == speaker_b:
            speaker_b_chunks.append((chunk, emb, idx))
        elif speaker == "mixed":
            speaker_a_chunks.append((chunk, emb, idx))
            speaker_b_chunks.append((chunk, emb, idx))

    def retrieve_for_speaker(speaker_chunks):
        """Rank chunks for one speaker."""
        if not speaker_chunks:
            return []

        results = []
        for chunk, emb, original_idx in speaker_chunks:
            norm_q = np.linalg.norm(question_embedding)
            norm_c = np.linalg.norm(emb)
            if norm_q > 0 and norm_c > 0:
                similarity = np.dot(question_embedding, emb) / (norm_q * norm_c)
            else:
                similarity = 0.0
            results.append((chunk, float(similarity), original_idx))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [(chunk, score, idx) for chunk, score, idx in results[:top_k]]

    if target_speaker == "speaker_a":
        speaker_a_results = retrieve_for_speaker(speaker_a_chunks)
        speaker_b_results = []
    elif target_speaker == "speaker_b":
        speaker_a_results = []
        speaker_b_results = retrieve_for_speaker(speaker_b_chunks)
    else:
        speaker_a_results = retrieve_for_speaker(speaker_a_chunks)
        speaker_b_results = retrieve_for_speaker(speaker_b_chunks)
    
    return speaker_a_results, speaker_b_results


def format_memories_for_speaker(
    retrieved_chunks: List[Tuple[Dict, float, int]],
    chunk_embeddings: List[np.ndarray]
) -> Tuple[str, List[TurnNode]]:
    """Format retrieved chunks and build TurnNode objects."""
    if not retrieved_chunks:
        return "No memories available.", []
    
    formatted_memories = []
    turn_nodes = []
    
    for idx, (chunk, score, original_idx) in enumerate(retrieved_chunks):
        date_str = chunk.get("date", "")
        text = chunk.get("text", "")
        
        formatted_date = date_str

        memory_entry = f"{idx + 1}. [Timestamp: {formatted_date}]\n{text}"
        formatted_memories.append(memory_entry)

        session_idx = int(chunk["session_key"].replace("session_", "")) if "session_" in chunk.get("session_key", "") else 0
        node = TurnNode(
            turn_id=f"session_{session_idx}_turn_{idx}",
            text=text,
            embedding=chunk_embeddings[original_idx] if original_idx < len(chunk_embeddings) else np.zeros(1536),
            date=date_str,
            session_idx=session_idx,
            turn_idx=idx,
            role=chunk.get("speaker", "user"),
            similarity_to_query=score
        )
        turn_nodes.append(node)
    
    return "\n\n".join(formatted_memories), turn_nodes


def build_speaker_separated_prompt(
    question: str,
    speaker_a_name: str,
    speaker_a_memories: str,
    speaker_b_name: str,
    speaker_b_memories: str,
    target_speaker: str = "both"
) -> Tuple[str, str]:
    """Build the prompt from speaker-specific memories."""
    if target_speaker == "speaker_a":
        memories = f"## Memories for {speaker_a_name}:\n\n{speaker_a_memories}"
    elif target_speaker == "speaker_b":
        memories = f"## Memories for {speaker_b_name}:\n\n{speaker_b_memories}"
    else:
        memories = f"## Memories for {speaker_a_name}:\n\n{speaker_a_memories}\n\n## Memories for {speaker_b_name}:\n\n{speaker_b_memories}"
    
    user_prompt = USER_PROMPT_LOCOMO.format(
        memories=memories,
        question=question
    )
    return SYSTEM_PROMPT_LOCOMO, user_prompt

async def precompute_sample_embeddings(
    samples: List[Dict],
    embedding_client: AsyncOpenAIEmbedding,
    chunk_mode: str = "turn"
) -> Dict[int, Dict]:
    """Precompute unit-level embeddings for each sample."""
    print(f"\n{'='*60}")
    print(f"预处理样本 Embeddings: {len(samples)} 个样本")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    precomputed_samples = {}
    
    all_chunks_info = []
    for sample_idx, sample in enumerate(samples, 1):
        conversation = sample.get('conversation', {})
        chunks, speaker_a, speaker_b = chunk_locomo_conversations_with_speaker(conversation, mode=chunk_mode)
        if chunks:
            all_chunks_info.append((sample_idx, chunks, speaker_a, speaker_b))
            print(f"  样本 {sample_idx}: {len(chunks)} 个 chunks")
    
    all_texts = []
    sample_text_ranges = {}
    chunk_unit_info = {}
    
    for sample_idx, chunks, speaker_a, speaker_b in all_chunks_info:
        sample_start = len(all_texts)
        unit_info = []
        for chunk in chunks:
            unit_start = len(all_texts)
            units = _split_text(chunk.get("raw_text", chunk["text"]))
            all_texts.extend(units)
            unit_end = len(all_texts)
            unit_info.append((unit_start, unit_end))
        sample_end = len(all_texts)
        sample_text_ranges[sample_idx] = (sample_start, sample_end)
        chunk_unit_info[sample_idx] = unit_info
    
    print(f"\n  开始批量计算 embeddings...")
    
    all_embeddings = await embedding_client.get_embeddings(all_texts, batch_size=20)
    
    for sample_idx, chunks, speaker_a, speaker_b in all_chunks_info:
        unit_info = chunk_unit_info[sample_idx]
        chunk_embeddings = []
        for unit_start, unit_end in unit_info:
            unit_embs = [np.array(e) for e in all_embeddings[unit_start:unit_end]]
            chunk_embeddings.append(unit_embs)
        precomputed_samples[sample_idx] = {
            "chunks": chunks,
            "embeddings": chunk_embeddings,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b
        }
    
    elapsed = time.time() - start_time
    print(f"\n  预处理完成! 耗时: {elapsed:.2f}s")
    print(f"{'='*60}\n")
    
    return precomputed_samples

async def run_retrieval_for_item_speaker(
    item: Dict,
    embedding_client: AsyncOpenAIEmbedding,
    top_k: int,
    precomputed_samples: Dict[int, Dict] = None
) -> Dict:
    """Run speaker-aware retrieval for one question."""
    question_id = item["question_id"]
    question = item["question"]
    sample_idx = item["sample_idx"]
    
    try:
        if precomputed_samples and sample_idx in precomputed_samples:
            chunks = precomputed_samples[sample_idx]["chunks"]
            raw_embeddings = precomputed_samples[sample_idx]["embeddings"]
            speaker_a = precomputed_samples[sample_idx]["speaker_a"]
            speaker_b = precomputed_samples[sample_idx]["speaker_b"]
            question_embeddings = await embedding_client.get_embeddings([question], batch_size=1)
            question_embedding = np.array(question_embeddings[0])
            chunk_embeddings = []
            for unit_embs in raw_embeddings:
                if not unit_embs:
                    chunk_embeddings.append(np.zeros(1536))
                    continue
                best_emb = unit_embs[0]
                best_sim = -1.0
                nq = np.linalg.norm(question_embedding)
                for emb in unit_embs:
                    ne = np.linalg.norm(emb)
                    if nq > 0 and ne > 0:
                        sim = np.dot(question_embedding, emb) / (nq * ne)
                        if sim > best_sim:
                            best_sim = sim
                            best_emb = emb
                chunk_embeddings.append(best_emb)
        else:
            chunks = item["chunks"]
            speaker_a = item.get("speaker_a", "Speaker_A")
            speaker_b = item.get("speaker_b", "Speaker_B")
            all_units = []
            chunk_unit_ranges = []
            for c in chunks:
                start = len(all_units)
                units = _split_text(c.get("raw_text", c["text"]))
                all_units.extend(units)
                end = len(all_units)
                chunk_unit_ranges.append((start, end))
            embeddings = await embedding_client.get_embeddings([question] + all_units, batch_size=200)
            question_embedding = np.array(embeddings[0])
            unit_embeddings = [np.array(e) for e in embeddings[1:]]
            chunk_embeddings = []
            nq = np.linalg.norm(question_embedding)
            for start, end in chunk_unit_ranges:
                unit_embs = unit_embeddings[start:end]
                if not unit_embs:
                    chunk_embeddings.append(np.zeros(1536))
                    continue
                best_emb = unit_embs[0]
                best_sim = -1.0
                for emb in unit_embs:
                    ne = np.linalg.norm(emb)
                    if nq > 0 and ne > 0:
                        sim = np.dot(question_embedding, emb) / (nq * ne)
                        if sim > best_sim:
                            best_sim = sim
                            best_emb = emb
                chunk_embeddings.append(best_emb)
        
        target_speaker = detect_target_speaker(question, speaker_a, speaker_b)

        speaker_a_results, speaker_b_results = retrieve_relevant_chunks_by_speaker(
            question_embedding, chunks, chunk_embeddings,
            speaker_a, speaker_b, top_k,
            target_speaker=target_speaker
        )
        
        speaker_a_memories, speaker_a_nodes = format_memories_for_speaker(speaker_a_results, chunk_embeddings)
        speaker_b_memories, speaker_b_nodes = format_memories_for_speaker(speaker_b_results, chunk_embeddings)

        all_turn_nodes = speaker_a_nodes + speaker_b_nodes
        
        return {
            "question_id": question_id,
            "question": question,
            "answer": item["answer"],
            "category": item["category"],
            "sample_idx": item["sample_idx"],
            "question_embedding": question_embedding.tolist(),
            "turn_nodes": all_turn_nodes,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "speaker_a_memories": speaker_a_memories,
            "speaker_b_memories": speaker_b_memories,
            "speaker_a_count": len(speaker_a_results),
            "speaker_b_count": len(speaker_b_results),
            "target_speaker": target_speaker,
            "success": True
        }
    except Exception as e:
        print(f"  [{question_id}] 检索错误: {e}")
        return {"question_id": question_id, "success": False, "error": str(e)}

async def process_item_with_com_speaker(
    retrieval_item: Dict,
    llm_client: AsyncOpenAILLM,
    embedding_client: AsyncOpenAIEmbedding,
    blocking_ratio: float,
    num_anchors: int,
    total_count: int,
    idx: int
) -> Dict:
    """Run CoM filtering and answer generation for one question."""
    question_id = retrieval_item["question_id"]
    
    if not retrieval_item.get("success", False):
        return {
            "result": {"question_id": question_id, "error": retrieval_item.get("error", "Retrieval failed")},
            "stats": {"success": False}
        }
    
    print(f"\n[{idx+1}/{total_count}] 处理: {question_id}")
    start_time = time.time()
    
    try:
        question = retrieval_item["question"]
        speaker_a = retrieval_item.get("speaker_a", "Speaker_A")
        speaker_b = retrieval_item.get("speaker_b", "Speaker_B")
        target_speaker = retrieval_item.get("target_speaker", "both")
        
        com_time = 0.0
        main_chain_size = 0
        num_chains = 0
        
        speaker_a_memories = "No memories available."
        speaker_b_memories = "No memories available."

        if "question_embedding" not in retrieval_item or "turn_nodes" not in retrieval_item:
            speaker_a_memories = retrieval_item.get("speaker_a_memories", "No memories available.")
            speaker_b_memories = retrieval_item.get("speaker_b_memories", "No memories available.")
            print(f"  [{question_id}] 警告: 缺少 turn_nodes，使用缓存的 memories")
        else:
            question_embedding = np.array(retrieval_item["question_embedding"])
            turn_nodes = retrieval_item["turn_nodes"]
            
            if target_speaker == "speaker_a":
                turn_nodes = [n for n in turn_nodes if speaker_a in n.text or n.text.startswith(f"{speaker_a}:")]
            elif target_speaker == "speaker_b":
                turn_nodes = [n for n in turn_nodes if speaker_b in n.text or n.text.startswith(f"{speaker_b}:")]

            if not turn_nodes:
                print(f"  [{question_id}] 警告: 过滤后没有 turn_nodes")
            else:
                start_com = time.time()
                com_result = await com_filter_async(
                    candidate_nodes=turn_nodes,
                    query_embedding=question_embedding,
                    blocking_ratio=blocking_ratio,
                    num_anchors=num_anchors,
                    sampling_method="top_k",
                    context_mode="concatenated",
                    embedding_client=embedding_client
                )
                com_time = time.time() - start_com
                main_chain_size = len(com_result.main_chain)
                num_chains = len(com_result.evidence_chains)

                def format_evidence_chains(chains):
                    """Format evidence chains for the prompt."""
                    if not chains:
                        return "No memories available."

                    formatted_parts = []
                    for chain_idx, chain in enumerate(chains, 1):
                        chain_lines = [f"=== Memory Chain {chain_idx} ==="]
                        for node_idx, node in enumerate(chain.nodes, 1):
                            date_str = node.date if node.date else ""
                            chain_lines.append(f"[{node_idx}] [Timestamp: {date_str}] {node.text}")
                        formatted_parts.append("\n".join(chain_lines))
                    
                    return "\n\n".join(formatted_parts) if formatted_parts else "No memories available."
                
                if target_speaker == "speaker_a":
                    speaker_a_memories = format_evidence_chains(com_result.evidence_chains)
                elif target_speaker == "speaker_b":
                    speaker_b_memories = format_evidence_chains(com_result.evidence_chains)
                else:
                    speaker_a_chains = []
                    speaker_b_chains = []

                    for chain in com_result.evidence_chains:
                        a_count = sum(1 for n in chain.nodes if speaker_a in n.text or n.text.startswith(f"{speaker_a}:"))
                        b_count = sum(1 for n in chain.nodes if speaker_b in n.text or n.text.startswith(f"{speaker_b}:"))
                        if a_count >= b_count:
                            speaker_a_chains.append(chain)
                        else:
                            speaker_b_chains.append(chain)
                    
                    speaker_a_memories = format_evidence_chains(speaker_a_chains)
                    speaker_b_memories = format_evidence_chains(speaker_b_chains)
                
                print(f"  [{question_id}] CoM: {com_time:.3f}s")
        
        system_prompt, user_prompt = build_speaker_separated_prompt(
            question=question,
            speaker_a_name=speaker_a,
            speaker_a_memories=speaker_a_memories,
            speaker_b_name=speaker_b,
            speaker_b_memories=speaker_b_memories,
            target_speaker=target_speaker
        )
        
        start_llm = time.time()
        response, usage = await llm_client.generate([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], return_usage=True)
        response = remove_think_tags(response)
        llm_time = time.time() - start_llm
        
        total_time = time.time() - start_time
        print(f"  [{question_id}] LLM: {llm_time:.2f}s, total: {total_time:.2f}s")
        
        if target_speaker == "speaker_a":
            retrieved_context = f"=== {speaker_a} ===\n{speaker_a_memories}"
        elif target_speaker == "speaker_b":
            retrieved_context = f"=== {speaker_b} ===\n{speaker_b_memories}"
        else:
            retrieved_context = f"=== {speaker_a} ===\n{speaker_a_memories}\n\n=== {speaker_b} ===\n{speaker_b_memories}"
        
        return {
            "result": {
                "sample_id": retrieval_item["sample_idx"],
                "question_id": question_id,
                "question": question,
                "prediction": response,
                "reference": retrieval_item["answer"],
                "category": retrieval_item["category"],
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "target_speaker": target_speaker,
                "retrieved_context": retrieved_context
            },
            "stats": {
                "tokens": usage.total_tokens if usage else 0,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "time": total_time,
                "com_time": com_time,
                "main_chain_size": main_chain_size,
                "num_chains": num_chains,
                "success": True
            }
        }
        
    except Exception as e:
        print(f"  [{question_id}] 错误: {e}")
        import traceback
        traceback.print_exc()
        return {
            "result": {"question_id": question_id, "error": str(e)},
            "stats": {"tokens": 0, "time": time.time() - start_time, "success": False}
        }


async def run_full_pipeline(
    items: List[Dict],
    llm_client: AsyncOpenAILLM,
    embedding_client: AsyncOpenAIEmbedding,
    output_file: str,
    top_k: int,
    blocking_ratio: float,
    num_anchors: int,
    max_concurrent: int,
    precomputed_samples: Dict[int, Dict] = None
) -> None:
    """Run the full LoCoMo pipeline."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"完整流程模式: {len(items)} 个问题")
    print(f"参数: top_k={top_k}")
    print(f"CoM参数: anchors={num_anchors}, blocking={blocking_ratio}, sampling=top_k")
    if precomputed_samples:
        print(f"使用预计算的 embeddings ({len(precomputed_samples)} 个样本)")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_full(idx, item):
        async with semaphore:
            retrieval_item = await run_retrieval_for_item_speaker(item, embedding_client, top_k, precomputed_samples)
            
            if not retrieval_item.get("success"):
                return {
                    "result": {"question_id": item["question_id"], "error": retrieval_item.get("error")},
                    "stats": {"success": False}
                }
            
            return await process_item_with_com_speaker(
                retrieval_item, llm_client, embedding_client,
                blocking_ratio, num_anchors, len(items), idx
            )
    
    tasks = [process_full(i, item) for i, item in enumerate(items)]
    all_results = await asyncio.gather(*tasks)

    results = [r["result"] for r in all_results]
    success_stats = [r["stats"] for r in all_results if r["stats"].get("success")]
    n_success = len(success_stats)
    total_tokens = sum(s["tokens"] for s in success_stats)
    total_com_time = sum(s.get("com_time", 0) for s in success_stats)
    
    category_counts = {}
    for r in all_results:
        if r["stats"].get("success"):
            cat = r["result"].get("category")
            if cat is not None:
                category_counts[str(cat)] = category_counts.get(str(cat), 0) + 1
    
    elapsed = time.time() - start_time
    
    final = {
        "model": llm_client.model,
        "method": "com_speaker_separated_full",
        "com_params": {
            "top_k": top_k, 
            "num_anchors": num_anchors, 
            "blocking_ratio": blocking_ratio, 
            "sampling_method": "top_k",
            "context_mode": "concatenated"
        },
        "retrieval_mode": "speaker_separated",
        "total_questions": len(items),
        "success_count": n_success,
        "error_count": len(items) - n_success,
        "category_distribution": category_counts,
        "token_statistics": {
            "total_tokens_k": round(total_tokens / 1000, 2),
            "avg_tokens_k": round(total_tokens / n_success / 1000, 2) if n_success > 0 else 0
        },
        "time_statistics": {
            "wall_clock_time": round(elapsed, 2),
            "total_com_time": round(total_com_time, 2)
        },
        "individual_results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"完成! 结果: {output_file}")
    print(f"{'='*60}")
    print(f"总问题数: {len(items)}, 成功: {n_success}, 失败: {len(items) - n_success}")
    print(f"总Token数: {total_tokens/1000:.2f}k, 平均每题: {total_tokens/n_success/1000:.2f}k" if n_success > 0 else f"总Token数: {total_tokens/1000:.2f}k")
    print(f"墙钟时间: {elapsed:.2f}s ({elapsed/60:.2f}分钟)")
    print(f"{'='*60}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="CoM evaluation on LoComo")

    parser.add_argument("--dataset", type=str, default="dataset/locomo10.json")
    parser.add_argument("--output", type=str, default="results/locomo/ours_locomo.json")
    parser.add_argument("--config", type=str, default="config/config.yaml")

    parser.add_argument("--top_k", type=int, default=20)

    parser.add_argument("--num_anchors", type=int, default=3)
    parser.add_argument("--blocking_ratio", type=float, default=0.5)

    parser.add_argument("--max_concurrent", type=int, default=50)
    parser.add_argument("--max_concurrent_llm", type=int, default=50)
    parser.add_argument("--max_concurrent_embed", type=int, default=50)
    
    return parser.parse_args()


def create_clients(config: dict, args):
    """Create the LLM and embedding clients."""
    llm_cfg = config.get('llm', {})
    embed_cfg = config.get('embedding', {})
    
    llm_client = AsyncOpenAILLM(
        api_key=llm_cfg.get('api_key', ''),
        base_url=llm_cfg.get('base_url'),
        model=llm_cfg.get('model', 'gpt-4o-mini'),
        temperature=llm_cfg.get('temperature', 0.0),
        max_tokens=llm_cfg.get('max_tokens', 2000),
        timeout=llm_cfg.get('timeout', 300.0),
        max_concurrent=args.max_concurrent_llm
    )
    
    embedding_client = AsyncOpenAIEmbedding(
        api_key=embed_cfg.get('api_key'),
        base_url=embed_cfg.get('base_url'),
        model=embed_cfg.get('model', 'text-embedding-3-small'),
        timeout=embed_cfg.get('timeout', 300.0),
        max_concurrent=args.max_concurrent_embed
    )
    
    return llm_client, embedding_client


async def main_async():
    args = parse_args()
    config = load_config(args.config)
    chunk_mode = "turn"

    print(f"{'='*60}")
    print("CoM LoComo evaluation")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Top-K: {args.top_k}")
    print(
        "CoM: "
        f"num_anchors={args.num_anchors}, "
        f"blocking_ratio={args.blocking_ratio}, "
        "sampling=top_k"
    )
    print(f"{'='*60}")

    llm_client, embedding_client = create_clients(config, args)
    samples = load_locomo_dataset(args.dataset)
    items = convert_locomo_to_items_with_speaker(samples, mode=chunk_mode)

    precomputed_samples = await precompute_sample_embeddings(
        samples, embedding_client, chunk_mode=chunk_mode
    )

    await run_full_pipeline(
        items, llm_client, embedding_client, args.output,
        args.top_k, args.blocking_ratio, args.num_anchors,
        args.max_concurrent,
        precomputed_samples=precomputed_samples
    )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


