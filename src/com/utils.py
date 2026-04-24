"""Small utility helpers for CoM."""

import json
import re
from typing import Any, Dict, List

import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if isinstance(vec1, list):
        vec1 = np.array(vec1)
    if isinstance(vec2, list):
        vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    print(f"Loading dataset: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    return data


def remove_think_tags(text: str) -> str:
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r"<think>.*", "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip()


def parse_turn_text(text: str) -> List[Dict[str, str]]:
    lines = text.split("\n")
    turns = []
    current_role = None
    current_content = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("[Timestamp:"):
            continue

        if line.startswith("user:"):
            if current_role and current_content:
                turns.append({"role": current_role, "content": "\n".join(current_content)})
            current_role = "user"
            current_content = [line[5:].strip()]
        elif line.startswith("assistant:"):
            if current_role and current_content:
                turns.append({"role": current_role, "content": "\n".join(current_content)})
            current_role = "assistant"
            current_content = [line[10:].strip()]
        elif current_role:
            current_content.append(line)

    if current_role and current_content:
        turns.append({"role": current_role, "content": "\n".join(current_content)})

    return turns
