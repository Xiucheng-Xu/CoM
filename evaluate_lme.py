"""LLM-as-a-Judge evaluation for LongMemEval."""

import argparse
import json
import os
import re
import sys
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from src.llm.llm_client import OpenAILLM
from src.llm.prompt_judge import (
    EVAL_ABSTENTION_PROMPT,
    EVAL_BASIC_PROMPT,
    EVAL_KNOWLEDGE_UPDATE_PROMPT,
    EVAL_PREFERENCE_PROMPT,
    EVAL_TEMPORAL_REASONING_PROMPT,
)


def get_eval_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    """Return the evaluation prompt for a question type."""
    if abstention:
        return EVAL_ABSTENTION_PROMPT.format(question, answer, response)

    if task in ["single-session-user", "single-session-assistant", "multi-session"]:
        return EVAL_BASIC_PROMPT.format(question, answer, response)
    if task == "temporal-reasoning":
        return EVAL_TEMPORAL_REASONING_PROMPT.format(question, answer, response)
    if task == "knowledge-update":
        return EVAL_KNOWLEDGE_UPDATE_PROMPT.format(question, answer, response)
    if task == "single-session-preference":
        return EVAL_PREFERENCE_PROMPT.format(question, answer, response)

    raise NotImplementedError(f"Unsupported task type: {task}")


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load the YAML config file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load a JSONL file, or fall back to JSON."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)


def load_json(file_path: str) -> Any:
    """Load a JSON file, or fall back to JSONL."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return load_jsonl(file_path)


def normalize_text(text: str) -> str:
    """Normalize whitespace for string matching."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_evidence_recall(
    hypotheses: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute evidence recall from retrieved context."""
    qid2ref = {entry["question_id"]: entry for entry in references}

    recall_results = []
    successful_recalls = 0
    total_samples = 0

    for hyp_entry in hypotheses:
        qid = hyp_entry["question_id"]
        if qid not in qid2ref:
            continue

        ref_entry = qid2ref[qid]

        retrieved_text = ""
        if hyp_entry.get("context"):
            retrieved_text = hyp_entry["context"]
        elif hyp_entry.get("retrieved_text"):
            retrieved_text = hyp_entry["retrieved_text"]

        evidences = []
        answer_session_ids = ref_entry.get("answer_session_ids", [])
        haystack_sessions = ref_entry.get("haystack_sessions", [])
        haystack_session_ids = ref_entry.get("haystack_session_ids", [])

        for ans_sid in answer_session_ids:
            if ans_sid in haystack_session_ids:
                idx = haystack_session_ids.index(ans_sid)
                session = haystack_sessions[idx]
                for turn in session:
                    if turn.get("has_answer", False):
                        evidences.append(turn["content"])

        if not evidences:
            continue

        total_samples += 1
        retrieved_text_normalized = normalize_text(retrieved_text)
        missing_evidences = []

        for evidence in evidences:
            if normalize_text(evidence) not in retrieved_text_normalized:
                missing_evidences.append(evidence)

        all_recalled = len(missing_evidences) == 0
        if all_recalled:
            successful_recalls += 1

        recall_results.append(
            {
                "question_id": qid,
                "all_recalled": all_recalled,
                "total_evidences": len(evidences),
                "recalled_evidences": len(evidences) - len(missing_evidences),
                "missing_evidences": missing_evidences,
            }
        )

    recall_rate = successful_recalls / total_samples if total_samples > 0 else 0
    return {
        "recall_rate": recall_rate,
        "successful_recalls": successful_recalls,
        "total_samples": total_samples,
        "details": recall_results,
    }


def evaluate_results(
    hyp_file: str = "results/lme/ours.jsonl",
    ref_file: str = "dataset/longmemeval_oracle_aligned.json",
    config_path: str = "config/config.yaml",
    output_file: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Evaluate LongMemEval predictions and save summary metrics."""
    if output_file is None:
        file_dir = os.path.dirname(hyp_file)
        file_name = os.path.basename(hyp_file)
        file_prefix = os.path.splitext(file_name)[0]
        output_file = os.path.join(file_dir, f"{file_prefix}_eval.json")

    print("=" * 60)
    print("LLM as Judge 评估脚本")
    print("=" * 60)
    print()

    print("正在加载配置...")
    config = load_config(config_path)

    print("正在初始化Judge LLM客户端...")
    judge_config = config.get("judge", {})
    judge_llm = OpenAILLM(
        api_key=judge_config.get("api_key"),
        base_url=judge_config.get("base_url"),
        model=judge_config.get("model", "gpt-4o"),
        temperature=judge_config.get("temperature", 0.0),
        max_tokens=judge_config.get("max_tokens", 10),
    )

    print(f"正在加载假设文件: {hyp_file}")
    hypotheses = load_jsonl(hyp_file)
    print(f"加载了 {len(hypotheses)} 条假设")

    print(f"正在加载参考文件: {ref_file}")
    references = load_json(ref_file)
    print(f"加载了 {len(references)} 条参考")

    qid2qdata = {entry["question_id"]: entry for entry in references}
    qid2qtype = {entry["question_id"]: entry["question_type"] for entry in references}
    qtypes = set(qid2qtype.values())
    qtype2acc = {t: [] for t in qtypes}

    print("正在计算证据召回率...")
    evidence_recall_result = calculate_evidence_recall(hypotheses, references)
    print(
        f"证据召回率: {evidence_recall_result['recall_rate']:.4f} "
        f"({evidence_recall_result['successful_recalls']}/{evidence_recall_result['total_samples']})"
    )
    print()

    print("开始评估...")

    details = []
    for entry in tqdm(hypotheses, desc="评估进度"):
        if entry["question_id"] not in qid2qtype:
            print(f'警告: 跳过 {entry["question_id"]}，该问题不在参考数据中。')
            continue

        qtype = qid2qtype[entry["question_id"]]
        q = qid2qdata[entry["question_id"]]["question"]
        ans = qid2qdata[entry["question_id"]]["answer"]
        hyp = entry["hypothesis"]

        is_abstention = "_abs" in entry["question_id"]
        prompt = get_eval_prompt(qtype, q, ans, hyp, abstention=is_abstention)

        messages = [{"role": "user", "content": prompt}]
        eval_response = judge_llm.generate(messages)
        label = "yes" in eval_response.lower()

        detail_entry = {
            "question_id": entry["question_id"],
            "question_type": qtype,
            "question_date": qid2qdata[entry["question_id"]].get("question_date", ""),
            "question": q,
            "hypothesis": hyp,
            "answer": ans,
            "judge_label": label,
        }
        details.append(detail_entry)

        if verbose:
            print(json.dumps(detail_entry, indent=4, ensure_ascii=False), flush=True)

        qtype2acc[qtype].append(1 if label else 0)

    print()
    print("=" * 60)
    print("评估完成！")
    print("=" * 60)

    overall_acc = np.mean([1 if x["judge_label"] else 0 for x in details])
    print(f"总体准确率: {overall_acc:.4f}")

    print()
    print("各类型准确率:")

    type_order = [
        ("single-session-user", "Single-session-user"),
        ("single-session-assistant", "Single-session-assistant"),
        ("single-session-preference", "Single-session-preference"),
        ("multi-session", "Multi-session"),
        ("temporal-reasoning", "Temporal-reasoning"),
        ("knowledge-update", "Knowledge-update"),
    ]
    single_hop_types = [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
    ]

    raw_accuracy_by_type = {}
    for key, values in qtype2acc.items():
        if values:
            raw_accuracy_by_type[key] = {
                "accuracy": np.mean(values),
                "count": len(values),
            }

    single_hop_samples = []
    for question_type in single_hop_types:
        if question_type in qtype2acc:
            single_hop_samples.extend(qtype2acc[question_type])
    if single_hop_samples:
        raw_accuracy_by_type["single-hop"] = {
            "accuracy": np.mean(single_hop_samples),
            "count": len(single_hop_samples),
        }

    ordered_accuracy_by_type = {}
    for key, display_name in type_order:
        if key in raw_accuracy_by_type:
            acc_value = raw_accuracy_by_type[key]["accuracy"]
            count_value = raw_accuracy_by_type[key]["count"]
            ordered_accuracy_by_type[key] = {
                "accuracy": round(acc_value, 4),
                "count": count_value,
            }
            print(f"\t{display_name}: {acc_value:.4f} ({count_value} 个样本)")

        if key == "single-session-preference" and "single-hop" in raw_accuracy_by_type:
            sh_acc = raw_accuracy_by_type["single-hop"]["accuracy"]
            sh_count = raw_accuracy_by_type["single-hop"]["count"]
            ordered_accuracy_by_type["single-hop"] = {
                "accuracy": round(sh_acc, 4),
                "count": sh_count,
            }
            print(f"\t  -> Single-hop: {sh_acc:.4f} ({sh_count} 个样本)")

    for key, value in raw_accuracy_by_type.items():
        if key not in ordered_accuracy_by_type:
            ordered_accuracy_by_type[key] = {
                "accuracy": round(value["accuracy"], 4),
                "count": value["count"],
            }
            print(f'\t{key}: {value["accuracy"]:.4f} ({value["count"]} 个样本)')

    result_data = {
        "summary": {
            "overall_accuracy": round(overall_acc, 4),
            "total_samples": len(details),
            "accuracy_by_type": ordered_accuracy_by_type,
            "evidence_recall": {
                "recall_rate": round(evidence_recall_result["recall_rate"], 4),
                "successful_recalls": evidence_recall_result["successful_recalls"],
                "total_samples": evidence_recall_result["total_samples"],
            },
        },
        "details": details,
        "evidence_recall_details": evidence_recall_result["details"],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print()
    print("证据召回统计:")
    print(f"\t证据召回率: {evidence_recall_result['recall_rate']:.4f}")
    print(
        f"\t成功召回样本数: "
        f"{evidence_recall_result['successful_recalls']}/{evidence_recall_result['total_samples']}"
    )

    print()
    print(f"结果已保存到: {output_file}")

    return result_data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LongMemEval predictions using LLM as Judge")
    parser.add_argument("--input", "-i", type=str, default="results/lme/ours.jsonl")
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        default="dataset/longmemeval_oracle_aligned.json",
    )
    parser.add_argument("--config", "-c", type=str, default="config/config.yaml")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_results(
        hyp_file=args.input,
        ref_file=args.reference,
        config_path=args.config,
        output_file=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
