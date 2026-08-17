import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def count_and_sample_source(
    input_path: Path,
    output_file,
    source: str,
    target_tokens: int,
    tokenizer,
    seed: int,
):
    """
    从本地候选池中抽取指定 token 数的数据。

    当前实现按文档顺序流式读取，
    使用 reservoir-style 随机起始偏移会让大文件处理复杂很多。
    候选池本身在下载阶段已经 shuffle，
    因此这里直接顺序取即可。
    """
    rng = random.Random(
        seed
    )

    total_tokens = 0
    documents = 0

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            item = json.loads(
                line
            )

            text = item["text"]

            token_count = len(
                tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )

            if token_count == 0:
                continue

            output_file.write(
                json.dumps(
                    {
                        "text": text,
                        "source": source,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            total_tokens += (
                token_count
            )

            documents += 1

            if total_tokens >= target_tokens:
                break

    if total_tokens < target_tokens:
        raise RuntimeError(
            f"{source} pool 不足："
            f"需要 {target_tokens:,} tokens，"
            f"实际只有 {total_tokens:,}"
        )

    return {
        "tokens": total_tokens,
        "documents": documents,
    }


def sample_corpus(
    pool_root: Path,
    output_path: Path,
    tokenizer_path: Path,
    target_tokens: int,
    zh_ratio: float,
    en_ratio: float,
    code_ratio: float,
    seed: int,
):
    ratios = {
        "zh": zh_ratio,
        "en": en_ratio,
        "code": code_ratio,
    }

    ratio_sum = sum(
        ratios.values()
    )

    if abs(
        ratio_sum - 1.0
    ) > 1e-6:
        raise ValueError(
            f"比例之和必须为 1，"
            f"当前为 {ratio_sum}"
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
        )
    )

    targets = {
        source: int(
            target_tokens * ratio
        )
        for source, ratio
        in ratios.items()
    }

    targets["en"] += (
        target_tokens
        - sum(targets.values())
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        output_path.unlink()

    print("=" * 72)
    print(
        "EppieLM Corpus Sampler"
    )
    print("=" * 72)
    print(
        f"Target: {target_tokens:,} tokens"
    )
    print(
        f"ZH:     {targets['zh']:,}"
    )
    print(
        f"EN:     {targets['en']:,}"
    )
    print(
        f"Code:   {targets['code']:,}"
    )

    results = {}

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        for index, source in enumerate(
            ["zh", "en", "code"]
        ):
            input_path = (
                pool_root
                / f"{source}.jsonl"
            )

            if not input_path.exists():
                raise FileNotFoundError(
                    f"候选池不存在: "
                    f"{input_path}"
                )

            print(
                f"\nSampling {source}..."
            )

            result = (
                count_and_sample_source(
                    input_path=input_path,
                    output_file=output_file,
                    source=source,
                    target_tokens=(
                        targets[source]
                    ),
                    tokenizer=tokenizer,
                    seed=seed + index,
                )
            )

            results[source] = result

            print(
                f"{source}: "
                f"{result['tokens']:,} tokens"
            )

    actual_tokens = sum(
        result["tokens"]
        for result
        in results.values()
    )

    meta = {
        "target_tokens": (
            target_tokens
        ),
        "actual_tokens": (
            actual_tokens
        ),
        "ratios": ratios,
        "sources": results,
        "seed": seed,
    }

    meta_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".meta.json"
        )
    )

    with meta_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            meta,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "Sampling finished"
    )
    print("=" * 72)
    print(
        f"Actual tokens: "
        f"{actual_tokens:,}"
    )
    print(
        f"Output: {output_path}"
    )
    print(
        f"Meta:   {meta_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample training corpus "
            "from local EppieLM pool"
        )
    )

    parser.add_argument(
        "--pool_root",
        type=str,
        default=r"E:\EppieLMData\pool",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
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
        "--target_tokens",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--zh_ratio",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--en_ratio",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--code_ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    sample_corpus(
        pool_root=Path(
            args.pool_root
        ),
        output_path=Path(
            args.output
        ),
        tokenizer_path=Path(
            args.tokenizer
        ),
        target_tokens=(
            args.target_tokens
        ),
        zh_ratio=args.zh_ratio,
        en_ratio=args.en_ratio,
        code_ratio=args.code_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()