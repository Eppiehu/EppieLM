import argparse
import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    temp_path.replace(path)


def tokenize_count(tokenizer, text: str) -> int:
    if not text:
        return 0

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def find_text_column(columns: list[str]) -> str:
    candidates = [
        "text",
        "content",
        "code",
    ]

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"Cannot find text column. Available columns: {columns}"
    )


def analyze_parquet_file(
    path: Path,
    tokenizer,
    batch_size: int,
) -> dict:
    parquet_file = pq.ParquetFile(path)

    text_column = find_text_column(
        parquet_file.schema.names
    )

    documents = 0
    characters = 0
    tokens = 0
    empty_documents = 0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=[text_column],
    ):
        values = batch.column(0).to_pylist()

        for text in values:
            if text is None:
                empty_documents += 1
                continue

            text = str(text)

            if not text.strip():
                empty_documents += 1
                continue

            document_tokens = tokenize_count(
                tokenizer,
                text,
            )

            documents += 1
            characters += len(text)
            tokens += document_tokens

    return {
        "path": str(path),
        "type": "parquet",
        "text_column": text_column,
        "documents": documents,
        "empty_documents": empty_documents,
        "characters": characters,
        "tokens": tokens,
        "file_bytes": path.stat().st_size,
    }


def detect_json_text_field(obj: dict) -> str:
    candidates = [
        "content",
        "text",
        "code",
    ]

    for candidate in candidates:
        if candidate in obj:
            return candidate

    raise ValueError(
        f"Cannot find text field. Available keys: {list(obj.keys())}"
    )


def analyze_json_gz_file(
    path: Path,
    tokenizer,
) -> dict:
    documents = 0
    characters = 0
    tokens = 0
    empty_documents = 0
    invalid_json = 0

    text_field = None

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue

            if text_field is None:
                text_field = detect_json_text_field(obj)

            text = obj.get(text_field)

            if text is None:
                empty_documents += 1
                continue

            text = str(text)

            if not text.strip():
                empty_documents += 1
                continue

            document_tokens = tokenize_count(
                tokenizer,
                text,
            )

            documents += 1
            characters += len(text)
            tokens += document_tokens

    return {
        "path": str(path),
        "type": "json.gz",
        "text_field": text_field,
        "documents": documents,
        "empty_documents": empty_documents,
        "invalid_json": invalid_json,
        "characters": characters,
        "tokens": tokens,
        "file_bytes": path.stat().st_size,
    }


def summarize_files(files: dict) -> dict:
    documents = 0
    empty_documents = 0
    invalid_json = 0
    characters = 0
    tokens = 0
    file_bytes = 0

    for result in files.values():
        documents += result.get("documents", 0)
        empty_documents += result.get(
            "empty_documents",
            0,
        )
        invalid_json += result.get(
            "invalid_json",
            0,
        )
        characters += result.get(
            "characters",
            0,
        )
        tokens += result.get(
            "tokens",
            0,
        )
        file_bytes += result.get(
            "file_bytes",
            0,
        )

    avg_tokens = (
        tokens / documents
        if documents
        else 0.0
    )

    avg_chars = (
        characters / documents
        if documents
        else 0.0
    )

    chars_per_token = (
        characters / tokens
        if tokens
        else 0.0
    )

    return {
        "files": len(files),
        "documents": documents,
        "empty_documents": empty_documents,
        "invalid_json": invalid_json,
        "characters": characters,
        "tokens": tokens,
        "avg_tokens_per_document": avg_tokens,
        "avg_chars_per_document": avg_chars,
        "chars_per_token": chars_per_token,
        "file_bytes": file_bytes,
        "file_gb": file_bytes / (1024 ** 3),
    }


def analyze_source(
    source_name: str,
    root: Path,
    pattern: str,
    tokenizer,
    stats_path: Path,
    batch_size: int,
) -> dict:
    files = sorted(
        root.rglob(pattern)
    )

    checkpoint = load_json(stats_path)

    completed = checkpoint.get(
        "files",
        {},
    )

    total_files = len(files)

    print()
    print(
        "=" * 70
    )
    print(
        f"{source_name}: {total_files} files"
    )
    print(
        "=" * 70
    )

    for index, path in enumerate(
        files,
        start=1,
    ):
        key = str(
            path.resolve()
        )

        if key in completed:
            result = completed[key]

            print(
                f"[{index}/{total_files}] "
                f"skip completed: {path.name} "
                f"tokens={result.get('tokens', 0):,}"
            )

            continue

        print()
        print(
            f"[{index}/{total_files}] "
            f"processing: {path}"
        )

        try:
            if path.name.endswith(
                ".json.gz"
            ):
                result = analyze_json_gz_file(
                    path,
                    tokenizer,
                )
            else:
                result = analyze_parquet_file(
                    path,
                    tokenizer,
                    batch_size,
                )

        except Exception as exc:
            print(
                f"[error] {path}"
            )
            print(
                repr(exc)
            )
            raise

        completed[key] = result

        checkpoint = {
            "source": source_name,
            "root": str(root),
            "files": completed,
            "summary": summarize_files(
                completed
            ),
        }

        save_json(
            stats_path,
            checkpoint,
        )

        print(
            f"documents={result['documents']:,}"
        )
        print(
            f"characters={result['characters']:,}"
        )
        print(
            f"tokens={result['tokens']:,}"
        )

        if result["documents"]:
            print(
                "avg tokens/doc="
                f"{result['tokens'] / result['documents']:.2f}"
            )

    final_summary = summarize_files(
        completed
    )

    checkpoint = {
        "source": source_name,
        "root": str(root),
        "files": completed,
        "summary": final_summary,
    }

    save_json(
        stats_path,
        checkpoint,
    )

    return final_summary


def build_summary(
    en_summary: dict,
    zh_summary: dict,
    code_summary: dict,
) -> dict:
    source_summaries = {
        "en": en_summary,
        "zh": zh_summary,
        "code": code_summary,
    }

    total_tokens = sum(
        item["tokens"]
        for item in source_summaries.values()
    )

    total_documents = sum(
        item["documents"]
        for item in source_summaries.values()
    )

    total_bytes = sum(
        item["file_bytes"]
        for item in source_summaries.values()
    )

    for name, item in source_summaries.items():
        item["token_ratio"] = (
            item["tokens"] / total_tokens
            if total_tokens
            else 0.0
        )

    return {
        "sources": source_summaries,
        "total_documents": total_documents,
        "total_tokens": total_tokens,
        "total_file_bytes": total_bytes,
        "total_file_gb": total_bytes / (1024 ** 3),
    }


def print_summary(summary: dict) -> None:
    print()
    print(
        "=" * 70
    )
    print(
        "FINAL CORPUS SUMMARY"
    )
    print(
        "=" * 70
    )

    for name in [
        "en",
        "zh",
        "code",
    ]:
        item = summary["sources"][name]

        print()
        print(
            name.upper()
        )

        print(
            f"files:      {item['files']:,}"
        )
        print(
            f"documents:  {item['documents']:,}"
        )
        print(
            f"tokens:     {item['tokens']:,}"
        )
        print(
            f"size:       {item['file_gb']:.2f} GB"
        )
        print(
            "token share:"
            f" {item['token_ratio'] * 100:.2f}%"
        )
        print(
            "avg tok/doc:"
            f" {item['avg_tokens_per_document']:.2f}"
        )

    print()
    print(
        "-" * 70
    )
    print(
        f"Total documents: {summary['total_documents']:,}"
    )
    print(
        f"Total tokens:    {summary['total_tokens']:,}"
    )
    print(
        f"Total size:      {summary['total_file_gb']:.2f} GB"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            "tokenizer/eppielm_tokenizer"
        ),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            r"E:\EppieLMData\raw"
        ),
    )

    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path(
            r"E:\EppieLMData\stats"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer_path = args.tokenizer.resolve()

    print(
        f"Tokenizer: {tokenizer_path}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )

    print(
        f"Tokenizer vocab size: {len(tokenizer):,}"
    )

    en_root = args.data_root / "en"
    zh_root = args.data_root / "zh"
    code_root = args.data_root / "code"

    args.stats_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    en_summary = analyze_source(
        source_name="en",
        root=en_root,
        pattern="*.parquet",
        tokenizer=tokenizer,
        stats_path=args.stats_dir / "en.json",
        batch_size=args.batch_size,
    )

    zh_summary = analyze_source(
        source_name="zh",
        root=zh_root,
        pattern="*.parquet",
        tokenizer=tokenizer,
        stats_path=args.stats_dir / "zh.json",
        batch_size=args.batch_size,
    )

    code_summary = analyze_source(
        source_name="code",
        root=code_root,
        pattern="*.json.gz",
        tokenizer=tokenizer,
        stats_path=args.stats_dir / "code.json",
        batch_size=args.batch_size,
    )

    summary = build_summary(
        en_summary=en_summary,
        zh_summary=zh_summary,
        code_summary=code_summary,
    )

    summary_path = (
        args.stats_dir
        / "summary.json"
    )

    save_json(
        summary_path,
        summary,
    )

    print_summary(
        summary
    )

    print()
    print(
        f"Summary saved to: {summary_path}"
    )


if __name__ == "__main__":
    main()