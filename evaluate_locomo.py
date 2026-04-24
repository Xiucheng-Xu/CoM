"""LLM-as-a-Judge evaluation for LoCoMo."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from src.llm.llm_client import OpenAILLM
from src.llm.prompt_judge import LOCOMO_JUDGE_PROMPT


def normalize_text(text: str) -> str:
    """Normalize whitespace for string matching."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load the YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_results(results_path: str) -> dict[str, Any] | list[dict[str, Any]]:
    """Load model outputs from JSON."""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(dataset_path: str = "dataset/locomo10.json") -> list[dict[str, Any]]:
    """Load the LoCoMo dataset."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_question_map(dataset: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map each question to its metadata and sessions."""
    question_map = {}

    for sample_idx, sample in enumerate(dataset, 1):
        conversation = sample.get("conversation", {})

        sessions = {}
        for key, value in conversation.items():
            if key.startswith("session_") and not key.endswith("_date_time"):
                session_num = key.replace("session_", "")
                sessions[f"D{session_num}"] = value

        for qa_idx, qa_item in enumerate(sample.get("qa", []), 1):
            question = qa_item.get("question", "").strip()
            evidences = qa_item.get("evidence", [])
            question_map[question] = {
                "sample_idx": sample_idx,
                "qa_idx": qa_idx,
                "evidences": evidences,
                "sessions": sessions,
            }
    return question_map


def create_judge_prompt(question: str, prediction: str, reference: str) -> str:
    """Create the judge prompt."""
    return LOCOMO_JUDGE_PROMPT.format(question, reference, prediction)


def calculate_evidence_recall(results: dict[str, Any], dataset: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute evidence recall from retrieved context."""
    question_map = build_question_map(dataset)

    recall_results = []
    successful_recalls = 0
    total_samples = 0

    for result_item in results.get("individual_results", []):
        if str(result_item.get("category", "")) == "5":
            continue

        question = result_item.get("question", "").strip()
        if question not in question_map:
            continue

        data = question_map[question]
        sample_idx = data["sample_idx"]
        qa_idx = data["qa_idx"]
        evidences = data["evidences"]
        sessions = data["sessions"]

        total_samples += 1

        context = (
            result_item.get("context", "")
            or result_item.get("retrieved_text", "")
            or result_item.get("retrieved_context", "")
        )

        evidence_texts = []
        for evidence_id in evidences:
            for part in evidence_id.replace(" ", "").split(";"):
                part = part.strip()
                if ":" not in part:
                    continue

                session_id, turn_idx_str = part.split(":", 1)
                try:
                    turn_idx = int(turn_idx_str) - 1
                    if session_id in sessions and 0 <= turn_idx < len(sessions[session_id]):
                        turn_text = sessions[session_id][turn_idx].get("text", "")
                        if turn_text:
                            evidence_texts.append(turn_text)
                except (ValueError, IndexError):
                    continue

        context_normalized = normalize_text(context)
        missing_evidences = [text[:100] for text in evidence_texts if normalize_text(text) not in context_normalized]
        all_recalled = len(missing_evidences) == 0 and len(evidence_texts) > 0

        if all_recalled:
            successful_recalls += 1

        recall_results.append(
            {
                "sample_id": f"sample_{sample_idx}_qa_{qa_idx}",
                "all_recalled": all_recalled,
                "total_evidences": len(evidence_texts),
                "recalled_evidences": len(evidence_texts) - len(missing_evidences),
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


def judge_single_prediction(
    judge_llm: OpenAILLM,
    question: str,
    prediction: str,
    reference: str,
    debug: bool = False,
) -> tuple[bool, str]:
    """Judge whether a prediction matches the reference."""
    prompt = create_judge_prompt(question, str(prediction), str(reference))
    messages = [{"role": "user", "content": prompt}]

    response = judge_llm.generate(messages, return_usage=False)
    raw_response = response.strip()

    clean_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
    response_upper = clean_response.upper()

    if debug:
        print(f"[DEBUG] Judge raw response: '{raw_response}'")
        print(f"[DEBUG] Judge cleaned response: '{clean_response}'")

    return response_upper.startswith("CORRECT"), clean_response


def evaluate_results(
    results_path: str,
    config_path: str = "config/config.yaml",
    output_path: str | None = None,
    sample_limit: int | None = None,
    dataset_path: str = "dataset/locomo10.json",
) -> dict[str, Any]:
    """Evaluate LoCoMo predictions and save summary metrics."""
    config = load_config(config_path)
    results = load_results(results_path)
    dataset = load_dataset(dataset_path)

    question_map = build_question_map(dataset)

    if isinstance(results, list):
        results = {"individual_results": results}
    elif isinstance(results, dict) and "individual_results" not in results:
        raise ValueError("Results dict does not contain 'individual_results' key")

    judge_config = config["judge"]
    judge_llm = OpenAILLM(
        api_key=judge_config["api_key"],
        base_url=judge_config.get("base_url"),
        model=judge_config["model"],
        temperature=judge_config.get("temperature", 0.0),
        max_tokens=judge_config.get("max_tokens", 4096),
    )

    print(f"Evaluating {len(results['individual_results'])} predictions...")

    print("\nCalculating evidence recall...")
    evidence_recall_result = calculate_evidence_recall(results, dataset)
    print(
        f"Evidence Recall Rate: {evidence_recall_result['recall_rate']:.4f} "
        f"({evidence_recall_result['successful_recalls']}/{evidence_recall_result['total_samples']})"
    )

    total_correct = 0
    total_samples = 0
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    detailed_results = []

    individual_results = results["individual_results"]
    if sample_limit:
        individual_results = individual_results[:sample_limit]
        print(f"Limiting evaluation to {sample_limit} samples")

    skipped_categories = defaultdict(int)
    for _, item in enumerate(tqdm(individual_results, desc="Evaluating"), 1):
        category_value = item.get("category")
        if category_value is None:
            skipped_categories["no_category"] += 1
            continue

        category = str(category_value)
        if category == "5":
            skipped_categories[category] += 1
            continue

        question = item.get("question", "")
        prediction = item.get("prediction", "")
        reference = item.get("reference", "")

        sample_id_str = str(item.get("sample_id", ""))
        mapped_info = question_map.get(question.strip())
        if mapped_info:
            sample_id_str = f"sample_{mapped_info['sample_idx']}_qa_{mapped_info['qa_idx']}"

        is_correct, judge_response = judge_single_prediction(
            judge_llm,
            question,
            prediction,
            reference,
            debug=False,
        )

        total_samples += 1
        category_stats[category]["total"] += 1

        if is_correct:
            total_correct += 1
            category_stats[category]["correct"] += 1

        detailed_results.append(
            {
                "sample_id": sample_id_str,
                "question": question,
                "prediction": prediction,
                "reference": reference,
                "category": category,
                "judge_result": "CORRECT" if is_correct else "INCORRECT",
                "judge_response": judge_response,
            }
        )

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    category_accuracy = {}
    for category, stats in category_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        category_accuracy[category] = {
            "accuracy": accuracy,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    eval_results = {
        "source_file": results_path,
        "total_samples": total_samples,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "category_accuracy": dict(sorted(category_accuracy.items())),
        "evidence_recall": {
            "recall_rate": round(evidence_recall_result["recall_rate"], 4),
            "successful_recalls": evidence_recall_result["successful_recalls"],
            "total_samples": evidence_recall_result["total_samples"],
        },
        "detailed_results": detailed_results,
        "evidence_recall_details": evidence_recall_result["details"],
    }

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Samples: {total_samples}")
    print(f"Total Correct: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
    if skipped_categories:
        skipped_msg = ", ".join(
            f"category {cat}: {count}"
            for cat, count in sorted(skipped_categories.items(), key=lambda x: (x[0].isdigit(), x[0]))
        )
        print(f"Skipped samples -> {skipped_msg}")
    print("\nAccuracy by Category:")
    print("-" * 60)
    for category in sorted(category_accuracy.keys(), key=int):
        cat_stats = category_accuracy[category]
        print(
            f"Category {category}: {cat_stats['accuracy']:.4f} ({cat_stats['accuracy'] * 100:.2f}%) "
            f"[{cat_stats['correct']}/{cat_stats['total']}]"
        )
    print("=" * 60)

    print("\nEvidence Recall Statistics:")
    print("-" * 60)
    print(
        f"Evidence Recall Rate: {evidence_recall_result['recall_rate']:.4f} "
        f"({evidence_recall_result['recall_rate'] * 100:.2f}%)"
    )
    print(
        f"Successful Recalls: "
        f"{evidence_recall_result['successful_recalls']}/{evidence_recall_result['total_samples']}"
    )
    print("=" * 60)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        print(f"\nEvaluation results saved to: {output_path}")

    return eval_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LoCoMo predictions using LLM as Judge")
    parser.add_argument("--input", "-i", type=str, default="results/locomo/ours_locomo.json")
    parser.add_argument("--config", "-c", type=str, default="config/config.yaml")

    args = parser.parse_args()

    results_path = Path(args.input)
    output_path = results_path.parent / f"{results_path.stem}_eval.json"

    evaluate_results(
        results_path=args.input,
        config_path=args.config,
        output_path=str(output_path),
        sample_limit=None,
    )


if __name__ == "__main__":
    main()
