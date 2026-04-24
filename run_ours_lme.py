import argparse
import asyncio

import yaml

from src.com import load_dataset, run_pipeline
from src.llm.async_embedding_client import AsyncOpenAIEmbedding
from src.llm.async_llm_client import AsyncOpenAILLM


def parse_args():
    parser = argparse.ArgumentParser(description="CoM evaluation on LongMemEval")

    parser.add_argument("--dataset", type=str, default="dataset/longmemeval_s_cleaned.json")
    parser.add_argument("--output", type=str, default="results/lme/ours.jsonl")
    parser.add_argument("--config", type=str, default="config/config.yaml")

    parser.add_argument("--top_k", type=int, default=20)

    parser.add_argument("--num_anchors", type=int, default=3)
    parser.add_argument("--blocking_ratio", type=float, default=0.5)

    parser.add_argument("--max_concurrent_samples", type=int, default=20)
    parser.add_argument("--max_concurrent_llm", type=int, default=20)
    parser.add_argument("--max_concurrent_embed", type=int, default=20)

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_clients(config: dict, args) -> tuple:
    llm_config = config.get("llm", {})
    embedding_config = config.get("embedding", {})

    llm_client = AsyncOpenAILLM(
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
        model=llm_config.get("model", "gpt-4o-mini"),
        temperature=llm_config.get("temperature", 0.0),
        max_tokens=llm_config.get("max_tokens", 8192),
        timeout=llm_config.get("timeout", 300.0),
        max_concurrent=args.max_concurrent_llm,
    )

    embedding_client = AsyncOpenAIEmbedding(
        api_key=embedding_config.get("api_key"),
        base_url=embedding_config.get("base_url"),
        model=embedding_config.get("model", "text-embedding-3-small"),
        timeout=embedding_config.get("timeout", 300.0),
        max_concurrent=args.max_concurrent_embed,
    )

    return llm_client, embedding_client


def print_config(args, config: dict):
    llm_config = config.get("llm", {})
    embedding_config = config.get("embedding", {})

    print(f"{'=' * 60}")
    print("CoM LongMemEval evaluation")
    print(f"{'=' * 60}")
    print(f"LLM: {llm_config.get('model', 'gpt-4o-mini')}")
    print(f"Embedding: {embedding_config.get('model', 'text-embedding-3-small')}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Top-K: {args.top_k}")
    print(
        "CoM: "
        f"num_anchors={args.num_anchors}, "
        f"blocking_ratio={args.blocking_ratio}, "
        "sampling=top_k, "
        "score_mode=multiply"
    )
    print(f"{'=' * 60}")


async def main_async():
    args = parse_args()
    config = load_config(args.config)
    llm_client, embedding_client = create_clients(config, args)
    print_config(args, config)

    output_file = args.output
    stats_file = output_file.replace(".jsonl", "_stats.json")

    await run_pipeline(
        data=load_dataset(args.dataset),
        llm_client=llm_client,
        embedding_client=embedding_client,
        output_file=output_file,
        stats_file=stats_file,
        top_k_turns=args.top_k,
        max_concurrent_samples=args.max_concurrent_samples,
        blocking_ratio=args.blocking_ratio,
        num_anchors=args.num_anchors,
    )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
