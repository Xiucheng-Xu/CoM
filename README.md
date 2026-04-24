# CoM

CoM is a lightweight research codebase for memory-augmented long-context question answering and evaluation.  
This repository provides a shared implementation of the CoM pipeline and runnable scripts for two benchmarks:

- `LongMemEval`
- `LoCoMo`

The codebase includes retrieval, chain construction, response generation, and LLM-as-a-Judge evaluation utilities.

## Features

- Shared CoM implementation under `src/com`
- Asynchronous LLM and embedding clients with simple caching
- End-to-end runners for LongMemEval and LoCoMo
- Evaluation scripts with evidence recall and judge-based accuracy

## Repository Structure

```text
.
├── config/
├── dataset/
├── results/
├── src/
│   ├── com/
│   └── llm/
├── run_ours_lme.py
├── run_ours_locomo.py
├── evaluate_lme.py
├── evaluate_locomo.py
└── requirements.txt
```

## Installation

Recommended environment: Python 3.12.12.

```bash
pip install -r requirements.txt
```

## Configuration

Create a local config file from the example:

```bash
cp config/config_example.yaml config/config.yaml
```

Then fill in the fields for:

- `llm`
- `embedding`
- `judge`

The clients are OpenAI-compatible, so both official OpenAI endpoints and compatible local / self-hosted services can be used.

## Running Experiments

### LongMemEval

```bash
python run_ours_lme.py \
  --dataset dataset/longmemeval_s_cleaned.json \
  --output results/lme/ours.jsonl \
  --config config/config.yaml \
  --top_k 20 \
  --num_anchors 3 \
  --blocking_ratio 0.5
```

Default outputs:

- `results/lme/ours.jsonl`
- `results/lme/ours_stats.json`

### LoCoMo

```bash
python run_ours_locomo.py \
  --dataset dataset/locomo10.json \
  --output results/locomo/ours_locomo.json \
  --config config/config.yaml \
  --top_k 20 \
  --num_anchors 3 \
  --blocking_ratio 0.5
```

Default output:

- `results/locomo/ours_locomo.json`

## Evaluation

### LongMemEval

```bash
python evaluate_lme.py \
  --input results/lme/ours.jsonl \
  --reference dataset/longmemeval_oracle_aligned.json \
  --config config/config.yaml
```

If `--output` is not provided, the evaluation file is written next to the input file as `*_eval.json`.

### LoCoMo

```bash
python evaluate_locomo.py \
  --input results/locomo/ours_locomo.json \
  --config config/config.yaml
```

If `--output` is not provided, the evaluation file is written next to the input file as `*_eval.json`.

## Notes

- `test_llm.py` is a small local connectivity test for the LLM and embedding endpoints.
