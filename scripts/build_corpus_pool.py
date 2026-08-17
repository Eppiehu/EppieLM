import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


SOURCES = {
    "zh": {
        "dataset": "HuggingFaceFW/fineweb-2",
        "subset": "cmn_Hani",
        "field": "text",
    },
    "en": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "field": "text",
    },
    "code": {
        "dataset": "codeparrot/codeparrot-clean",
        "subset": None,
        "field": "content",
    },
}


def configure_cache(
    cache_root: Path,
) -> None:
    cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.environ["HF_HOME"] = str(
        cache_root
    )

    os.environ["HF_HUB_CACHE"] = str(
        cache_root / "hub"
    )

    os.environ["HF_DATASETS_CACHE"] = str(
        cache_root / "datasets"
    )


def clean_text(
    text,
) -> str:
    if not isinstance(text, str):
        return ""

    return (
        text
        .replace("\x00", "")
        .strip()
    )


def valid_document(
    text: str,
    source: str,
) -> bool:
    if not text:
        return False

    if source == "code":
        return (
            200
            <= len(text)
            <= 200_000
        )

    return (
        200
        <= len(text)
        <= 500_000
    )


def build_source_pool(
    source: str,
    target_tokens: int,
    output_path: Path,
    tokenizer,
    seed: int,
) -> dict:
    info = SOURCES[source]

    kwargs = {
        "path": info["dataset"],
        "split": "train",
        "streaming": True,
    }

    if info["subset"] is not None:
        kwargs["name"] = (
            info["subset"]
        )

    print()
    print("=" * 72)
    print(
        f"Building source pool: {source}"
    )
    print("=" * 72)
    print(
        f"Dataset:       {info['dataset']}"
    )
    print(
        f"Subset:        {info['subset']}"
    )
    print(
        f"Target tokens: {target_tokens:,}"
    )
    print(
        f"Output:        {output_path}"
    )

    dataset = load_dataset(
        **kwargs
    )

    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=10_000,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        output_path.unlink()

    accepted_documents = 0
    rejected_documents = 0
    total_tokens = 0

    start_time = time.time()

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for item in dataset:
            text = clean_text(
                item.get(
                    info["field"],
                    "",
                )
            )

            if not valid_document(
                text,
                source,
            ):
                rejected_documents += 1
                continue

            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            num_tokens = len(
                token_ids
            )

            if num_tokens == 0:
                rejected_documents += 1
                continue

            record = {
                "text": text,
                "source": source,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            total_tokens += num_tokens
            accepted_documents += 1

            if (
                accepted_documents
                % 1000
                == 0
            ):
                elapsed = (
                    time.time()
                    - start_time
                )

                speed = (
                    total_tokens
                    / elapsed
                    if elapsed > 0
                    else 0
                )

                progress = min(
                    total_tokens
                    / target_tokens,
                    1.0,
                )

                print(
                    f"\r"
                    f"docs={accepted_documents:,} | "
                    f"tokens={total_tokens:,}/"
                    f"{target_tokens:,} | "
                    f"{progress:.2%} | "
                    f"tok/s={speed:,.0f}",
                    end="",
                    flush=True,
                )

            if total_tokens >= target_tokens:
                break

    elapsed = (
        time.time()
        - start_time
    )

    file_size_gb = (
        output_path.stat().st_size
        / 1024**3
    )

    print()
    print(
        f"Finished {source}: "
        f"{total_tokens:,} tokens, "
        f"{accepted_documents:,} docs"
    )

    return {
        "source": source,
        "dataset": info["dataset"],
        "subset": info["subset"],
        "target_tokens": target_tokens,
        "actual_tokens": total_tokens,
        "accepted_documents": (
            accepted_documents
        ),
        "rejected_documents": (
            rejected_documents
        ),
        "elapsed_seconds": elapsed,
        "file_size_gb": file_size_gb,
    }


def build_pool(
    output_root: Path,
    tokenizer_path: Path,
    cache_root: Path,
    zh_tokens: int,
    en_tokens: int,
    code_tokens: int,
    seed: int,
) -> None:
    configure_cache(
        cache_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
        )
    )

    targets = {
        "zh": zh_tokens,
        "en": en_tokens,
        "code": code_tokens,
    }

    print("=" * 72)
    print(
        "EppieLM Corpus Pool Builder"
    )
    print("=" * 72)
    print(
        f"Output root: {output_root}"
    )
    print(
        f"HF cache:    {cache_root}"
    )
    print(
        f"Tokenizer:   {tokenizer_path}"
    )
    print(
        f"Vocab:       {len(tokenizer):,}"
    )
    print()
    print(
        f"ZH target:   {zh_tokens:,}"
    )
    print(
        f"EN target:   {en_tokens:,}"
    )
    print(
        f"Code target: {code_tokens:,}"
    )
    print(
        f"Total:       "
        f"{sum(targets.values()):,}"
    )

    results = {}

    for index, source in enumerate(
        ["zh", "en", "code"]
    ):
        target = targets[source]

        if target <= 0:
            continue

        result = build_source_pool(
            source=source,
            target_tokens=target,
            output_path=(
                output_root
                / f"{source}.jsonl"
            ),
            tokenizer=tokenizer,
            seed=seed + index,
        )

        results[source] = result

    metadata = {
        "seed": seed,
        "tokenizer": str(
            tokenizer_path
        ),
        "total_target_tokens": (
            sum(targets.values())
        ),
        "total_actual_tokens": (
            sum(
                item["actual_tokens"]
                for item
                in results.values()
            )
        ),
        "sources": results,
    }

    metadata_path = (
        output_root
        / "pool_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "Corpus pool finished"
    )
    print("=" * 72)

    for source, result in (
        results.items()
    ):
        print(
            f"{source:5s}: "
            f"{result['actual_tokens']:,} "
            f"tokens | "
            f"{result['file_size_gb']:.2f} GB"
        )

    print()
    print(
        f"Metadata: {metadata_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build large reusable "
            "EppieLM corpus pool"
        )
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=r"E:\EppieLMData\pool",
    )

    parser.add_argument(
        "--cache_root",
        type=str,
        default=r"E:\EppieLMData\hf_cache",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=(
            "tokenizer/"
            "eppielm_tokenizer"
        ),
    )

    parser.add_argument(
        "--zh_tokens",
        type=int,
        default=14_000_000_000,
    )

    parser.add_argument(
        "--en_tokens",
        type=int,
        default=13_000_000_000,
    )

    parser.add_argument(
        "--code_tokens",
        type=int,
        default=3_000_000_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    build_pool(
        output_root=Path(
            args.output_root
        ),
        tokenizer_path=Path(
            args.tokenizer
        ),
        cache_root=Path(
            args.cache_root
        ),
        zh_tokens=args.zh_tokens,
        en_tokens=args.en_tokens,
        code_tokens=args.code_tokens,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()