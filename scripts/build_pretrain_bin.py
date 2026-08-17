import argparse
import gzip
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


VERSION = 1

SOURCE_ORDER = ("en", "zh", "code")

TEXT_FIELD_CANDIDATES = (
    "text",
    "content",
    "code",
)


def save_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(path.suffix + ".tmp")

    with temp.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(temp, path)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_text_field(names) -> str:
    names = list(names)

    for candidate in TEXT_FIELD_CANDIDATES:
        if candidate in names:
            return candidate

    raise ValueError(
        "Cannot find text/content/code field. "
        f"Available fields: {names}"
    )


def get_file_size_tokens(path: Path) -> int:
    if not path.exists():
        return 0

    size = path.stat().st_size

    if size % 2 != 0:
        raise RuntimeError(
            f"Invalid uint16 file size: {path}, bytes={size}"
        )

    return size // 2


def truncate_uint16_file(
    path: Path,
    token_count: int,
) -> None:
    expected_bytes = token_count * 2

    if not path.exists():
        if expected_bytes != 0:
            raise FileNotFoundError(path)
        return

    actual_bytes = path.stat().st_size

    if actual_bytes < expected_bytes:
        raise RuntimeError(
            f"Checkpoint expects {expected_bytes} bytes, "
            f"but {path} only has {actual_bytes} bytes."
        )

    if actual_bytes > expected_bytes:
        print(
            f"[resume] truncating {path.name}: "
            f"{actual_bytes // 2:,} -> {token_count:,} tokens"
        )

        with path.open("r+b") as f:
            f.truncate(expected_bytes)


def write_token_array(
    output_file,
    token_ids: list[int],
    remaining: int,
    vocab_size: int,
) -> int:
    if remaining <= 0 or not token_ids:
        return 0

    if len(token_ids) > remaining:
        token_ids = token_ids[:remaining]

    array = np.asarray(
        token_ids,
        dtype=np.uint16,
    )

    if array.size:
        max_id = int(array.max())

        if max_id >= vocab_size:
            raise ValueError(
                f"Token ID {max_id} >= vocab_size {vocab_size}"
            )

        array.tofile(output_file)

    return int(array.size)


class Progress:
    def __init__(
        self,
        source: str,
        initial_tokens: int,
        target_tokens: int,
    ):
        self.source = source
        self.initial_tokens = initial_tokens
        self.target_tokens = target_tokens

        self.start_time = time.perf_counter()
        self.last_print_time = self.start_time
        self.last_print_tokens = initial_tokens

    def maybe_print(
        self,
        current_tokens: int,
        force: bool = False,
    ) -> None:
        now = time.perf_counter()

        if not force and now - self.last_print_time < 15:
            return

        run_tokens = current_tokens - self.initial_tokens
        run_time = max(now - self.start_time, 1e-9)

        avg_rate = run_tokens / run_time

        recent_tokens = current_tokens - self.last_print_tokens
        recent_time = max(
            now - self.last_print_time,
            1e-9,
        )

        recent_rate = recent_tokens / recent_time

        ratio = (
            current_tokens / self.target_tokens
            if self.target_tokens
            else 1.0
        )

        remaining = max(
            self.target_tokens - current_tokens,
            0,
        )

        eta_seconds = (
            remaining / avg_rate
            if avg_rate > 0
            else float("inf")
        )

        if eta_seconds == float("inf"):
            eta_text = "unknown"
        else:
            eta_hours = eta_seconds / 3600
            eta_text = f"{eta_hours:.2f}h"

        print(
            f"[{self.source}] "
            f"{current_tokens / 1e9:.4f}B / "
            f"{self.target_tokens / 1e9:.4f}B "
            f"({ratio * 100:.2f}%) | "
            f"recent={recent_rate:,.0f} tok/s | "
            f"avg={avg_rate:,.0f} tok/s | "
            f"ETA={eta_text}"
        )

        self.last_print_time = now
        self.last_print_tokens = current_tokens


def tokenize_documents(
    tokenizer,
    texts: list[str],
    eos_token_id: int,
) -> list[list[int]]:
    if not texts:
        return []

    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    input_ids = encoded["input_ids"]

    # 每篇文档用 EOS 分隔，后续拼成连续预训练 token stream。
    for ids in input_ids:
        ids.append(eos_token_id)

    return input_ids


def append_document_batch(
    output_file,
    tokenizer,
    texts: list[str],
    eos_token_id: int,
    current_tokens: int,
    target_tokens: int,
    vocab_size: int,
) -> int:
    encoded_documents = tokenize_documents(
        tokenizer,
        texts,
        eos_token_id,
    )

    remaining = target_tokens - current_tokens

    if remaining <= 0:
        return current_tokens

    # 一个文档 batch 一次转换成 uint16，避免逐 token 写磁盘。
    flat_tokens = []

    for ids in encoded_documents:
        if not ids:
            continue

        available = remaining - len(flat_tokens)

        if available <= 0:
            break

        if len(ids) <= available:
            flat_tokens.extend(ids)
        else:
            flat_tokens.extend(
                ids[:available]
            )
            break

    written = write_token_array(
        output_file,
        flat_tokens,
        remaining,
        vocab_size,
    )

    return current_tokens + written


def checkpoint_source(
    state_path: Path,
    state: dict,
    source: str,
    file_index: int,
    resume_position: int,
    tokens_written: int,
) -> None:
    source_state = state["sources"][source]

    source_state["file_index"] = file_index
    source_state["resume_position"] = resume_position
    source_state["tokens_written"] = tokens_written

    save_json_atomic(
        state_path,
        state,
    )


def process_parquet_source(
    source: str,
    files: list[Path],
    output_path: Path,
    tokenizer,
    eos_token_id: int,
    vocab_size: int,
    target_tokens: int,
    batch_docs: int,
    state: dict,
    state_path: Path,
) -> None:
    source_state = state["sources"][source]

    file_index = int(
        source_state["file_index"]
    )

    resume_batch = int(
        source_state["resume_position"]
    )

    tokens_written = int(
        source_state["tokens_written"]
    )

    truncate_uint16_file(
        output_path,
        tokens_written,
    )

    progress = Progress(
        source,
        tokens_written,
        target_tokens,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("ab") as output_file:
        while (
            file_index < len(files)
            and tokens_written < target_tokens
        ):
            path = files[file_index]

            parquet_file = pq.ParquetFile(
                path
            )

            field = find_text_field(
                parquet_file.schema.names
            )

            print()
            print(
                f"[{source}] file "
                f"{file_index + 1}/{len(files)}: "
                f"{path}"
            )
            print(
                f"[{source}] text field: {field}"
            )

            for batch_index, batch in enumerate(
                parquet_file.iter_batches(
                    batch_size=batch_docs,
                    columns=[field],
                )
            ):
                if batch_index < resume_batch:
                    continue

                values = batch.column(0).to_pylist()

                texts = []

                for value in values:
                    if value is None:
                        continue

                    text = str(value)

                    if not text.strip():
                        continue

                    texts.append(text)

                if texts:
                    tokens_written = append_document_batch(
                        output_file=output_file,
                        tokenizer=tokenizer,
                        texts=texts,
                        eos_token_id=eos_token_id,
                        current_tokens=tokens_written,
                        target_tokens=target_tokens,
                        vocab_size=vocab_size,
                    )

                    output_file.flush()

                checkpoint_source(
                    state_path=state_path,
                    state=state,
                    source=source,
                    file_index=file_index,
                    resume_position=batch_index + 1,
                    tokens_written=tokens_written,
                )

                progress.maybe_print(
                    tokens_written
                )

                if tokens_written >= target_tokens:
                    break

            if tokens_written >= target_tokens:
                break

            file_index += 1
            resume_batch = 0

            checkpoint_source(
                state_path=state_path,
                state=state,
                source=source,
                file_index=file_index,
                resume_position=0,
                tokens_written=tokens_written,
            )

    progress.maybe_print(
        tokens_written,
        force=True,
    )

    if tokens_written < target_tokens:
        raise RuntimeError(
            f"{source} corpus exhausted at "
            f"{tokens_written:,} tokens, "
            f"target is {target_tokens:,}."
        )

    source_state["completed"] = True

    save_json_atomic(
        state_path,
        state,
    )


def process_code_source(
    source: str,
    files: list[Path],
    output_path: Path,
    tokenizer,
    eos_token_id: int,
    vocab_size: int,
    target_tokens: int,
    batch_docs: int,
    state: dict,
    state_path: Path,
) -> None:
    source_state = state["sources"][source]

    file_index = int(
        source_state["file_index"]
    )

    resume_line = int(
        source_state["resume_position"]
    )

    tokens_written = int(
        source_state["tokens_written"]
    )

    truncate_uint16_file(
        output_path,
        tokens_written,
    )

    progress = Progress(
        source,
        tokens_written,
        target_tokens,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("ab") as output_file:
        while (
            file_index < len(files)
            and tokens_written < target_tokens
        ):
            path = files[file_index]

            print()
            print(
                f"[{source}] file "
                f"{file_index + 1}/{len(files)}: "
                f"{path}"
            )

            texts = []
            last_line_number = resume_line

            with gzip.open(
                path,
                "rt",
                encoding="utf-8",
                errors="replace",
            ) as f:
                for line_number, line in enumerate(
                    f,
                    start=1,
                ):
                    if line_number <= resume_line:
                        continue

                    last_line_number = line_number

                    line = line.strip()

                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    field = None

                    for candidate in TEXT_FIELD_CANDIDATES:
                        if candidate in obj:
                            field = candidate
                            break

                    if field is None:
                        continue

                    value = obj.get(field)

                    if value is None:
                        continue

                    text = str(value)

                    if not text.strip():
                        continue

                    texts.append(text)

                    if len(texts) < batch_docs:
                        continue

                    tokens_written = append_document_batch(
                        output_file=output_file,
                        tokenizer=tokenizer,
                        texts=texts,
                        eos_token_id=eos_token_id,
                        current_tokens=tokens_written,
                        target_tokens=target_tokens,
                        vocab_size=vocab_size,
                    )

                    output_file.flush()

                    texts.clear()

                    checkpoint_source(
                        state_path=state_path,
                        state=state,
                        source=source,
                        file_index=file_index,
                        resume_position=last_line_number,
                        tokens_written=tokens_written,
                    )

                    progress.maybe_print(
                        tokens_written
                    )

                    if tokens_written >= target_tokens:
                        break

                if (
                    texts
                    and tokens_written < target_tokens
                ):
                    tokens_written = append_document_batch(
                        output_file=output_file,
                        tokenizer=tokenizer,
                        texts=texts,
                        eos_token_id=eos_token_id,
                        current_tokens=tokens_written,
                        target_tokens=target_tokens,
                        vocab_size=vocab_size,
                    )

                    output_file.flush()

                    checkpoint_source(
                        state_path=state_path,
                        state=state,
                        source=source,
                        file_index=file_index,
                        resume_position=last_line_number,
                        tokens_written=tokens_written,
                    )

                    progress.maybe_print(
                        tokens_written
                    )

            if tokens_written >= target_tokens:
                break

            file_index += 1
            resume_line = 0

            checkpoint_source(
                state_path=state_path,
                state=state,
                source=source,
                file_index=file_index,
                resume_position=0,
                tokens_written=tokens_written,
            )

    progress.maybe_print(
        tokens_written,
        force=True,
    )

    if tokens_written < target_tokens:
        raise RuntimeError(
            f"{source} corpus exhausted at "
            f"{tokens_written:,} tokens, "
            f"target is {target_tokens:,}."
        )

    source_state["completed"] = True

    save_json_atomic(
        state_path,
        state,
    )


def build_final_bin(
    work_paths: dict[str, Path],
    final_bin_path: Path,
    source_chunks: dict[str, int],
    seq_len: int,
    seed: int,
) -> None:
    final_part = final_bin_path.with_suffix(
        final_bin_path.suffix + ".part"
    )

    if final_part.exists():
        print(
            f"[assemble] removing incomplete file: {final_part}"
        )
        final_part.unlink()

    total_chunks = sum(
        source_chunks.values()
    )

    print()
    print(
        "=" * 72
    )
    print("ASSEMBLING FINAL TRAINING BIN")
    print("=" * 72)

    for source in SOURCE_ORDER:
        print(
            f"{source}: "
            f"{source_chunks[source]:,} chunks"
        )

    print(
        f"total: {total_chunks:,} chunks"
    )

    source_arrays = {}

    for source in SOURCE_ORDER:
        expected_tokens = (
            source_chunks[source] * seq_len
        )

        actual_tokens = get_file_size_tokens(
            work_paths[source]
        )

        if actual_tokens < expected_tokens:
            raise RuntimeError(
                f"{source}: only {actual_tokens:,} tokens "
                f"available, need {expected_tokens:,}."
            )

        source_arrays[source] = np.memmap(
            work_paths[source],
            mode="r",
            dtype=np.uint16,
            shape=(
                source_chunks[source],
                seq_len,
            ),
        )

    # 只随机来源顺序，不对巨型 token 数组整体加载到内存。
    labels = np.concatenate(
        [
            np.full(
                source_chunks["en"],
                0,
                dtype=np.uint8,
            ),
            np.full(
                source_chunks["zh"],
                1,
                dtype=np.uint8,
            ),
            np.full(
                source_chunks["code"],
                2,
                dtype=np.uint8,
            ),
        ]
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(labels)

    source_names = {
        0: "en",
        1: "zh",
        2: "code",
    }

    cursors = {
        "en": 0,
        "zh": 0,
        "code": 0,
    }

    assembly_batch_chunks = 2048

    start_time = time.perf_counter()

    with final_part.open("wb") as output_file:
        for start in range(
            0,
            total_chunks,
            assembly_batch_chunks,
        ):
            end = min(
                start + assembly_batch_chunks,
                total_chunks,
            )

            batch_labels = labels[start:end]

            output = np.empty(
                (
                    len(batch_labels),
                    seq_len,
                ),
                dtype=np.uint16,
            )

            for label, source in source_names.items():
                positions = np.flatnonzero(
                    batch_labels == label
                )

                count = len(positions)

                if count == 0:
                    continue

                cursor = cursors[source]

                output[positions] = (
                    source_arrays[source][
                        cursor:cursor + count
                    ]
                )

                cursors[source] += count

            output.tofile(output_file)

            completed = end

            if (
                completed == total_chunks
                or completed % 100_000 < assembly_batch_chunks
            ):
                elapsed = max(
                    time.perf_counter() - start_time,
                    1e-9,
                )

                rate = (
                    completed * seq_len
                    / elapsed
                )

                print(
                    f"[assemble] "
                    f"{completed:,}/{total_chunks:,} chunks "
                    f"({completed / total_chunks * 100:.2f}%) | "
                    f"{rate / 1e6:.2f}M tok/s"
                )

    os.replace(
        final_part,
        final_bin_path,
    )

    print(
        f"[assemble] saved: {final_bin_path}"
    )


def create_initial_state(
    tokenizer_path: Path,
    seq_len: int,
    requested_targets: dict[str, int],
    actual_targets: dict[str, int],
    seed: int,
) -> dict:
    return {
        "version": VERSION,
        "tokenizer": str(
            tokenizer_path.resolve()
        ),
        "seq_len": seq_len,
        "mix_seed": seed,
        "requested_targets": requested_targets,
        "actual_targets": actual_targets,
        "sources": {
            source: {
                "file_index": 0,
                "resume_position": 0,
                "tokens_written": 0,
                "completed": False,
            }
            for source in SOURCE_ORDER
        },
    }


def validate_state(
    state: dict,
    seq_len: int,
    actual_targets: dict[str, int],
) -> None:
    if state.get("version") != VERSION:
        raise RuntimeError(
            "Build state version mismatch."
        )

    if int(state["seq_len"]) != seq_len:
        raise RuntimeError(
            "seq_len differs from existing build state."
        )

    if state["actual_targets"] != actual_targets:
        raise RuntimeError(
            "Token targets differ from existing build state. "
            "Use the same targets or start a new output directory."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build EppieLM pretraining uint16 bin "
            "directly from local raw corpora."
        )
    )

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            "tokenizer/eppielm_tokenizer"
        ),
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            r"E:\EppieLMData\raw"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"E:\EppieLMData\pretrain"
        ),
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="eppielm_3b",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--en-tokens",
        type=int,
        default=1_350_000_000,
    )

    parser.add_argument(
        "--zh-tokens",
        type=int,
        default=1_350_000_000,
    )

    parser.add_argument(
        "--code-tokens",
        type=int,
        default=300_000_000,
    )

    parser.add_argument(
        "--batch-docs",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer_path = args.tokenizer.resolve()

    print(
        f"Tokenizer: {tokenizer_path}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )

    # 这里只做 tokenization，不把完整文档送入模型。
    # 原始文档超过 8192 很正常，后面才切成 2048-token chunks。
    tokenizer.model_max_length = 10**30

    vocab_size = len(tokenizer)

    if vocab_size > 65536:
        raise ValueError(
            f"vocab_size={vocab_size} does not fit uint16."
        )

    eos_token_id = tokenizer.eos_token_id

    if eos_token_id is None:
        raise ValueError(
            "Tokenizer has no eos_token_id."
        )

    print(
        f"Vocab size: {vocab_size:,}"
    )
    print(
        f"EOS token id: {eos_token_id}"
    )
    print(
        f"Sequence length: {args.seq_len:,}"
    )

    requested_targets = {
        "en": int(args.en_tokens),
        "zh": int(args.zh_tokens),
        "code": int(args.code_tokens),
    }

    # 每个来源都裁成完整 2048-token chunk。
    # 这样最终数据不会出现不完整训练样本。
    source_chunks = {
        source: (
            requested_targets[source]
            // args.seq_len
        )
        for source in SOURCE_ORDER
    }

    actual_targets = {
        source: (
            source_chunks[source]
            * args.seq_len
        )
        for source in SOURCE_ORDER
    }

    total_tokens = sum(
        actual_targets.values()
    )

    total_chunks = sum(
        source_chunks.values()
    )

    print()
    print(
        "=" * 72
    )
    print("TARGET")
    print("=" * 72)

    for source in SOURCE_ORDER:
        print(
            f"{source:>4}: "
            f"{actual_targets[source]:,} tokens | "
            f"{source_chunks[source]:,} chunks | "
            f"{actual_targets[source] / total_tokens * 100:.2f}%"
        )

    print(
        f"total: {total_tokens:,} tokens "
        f"({total_tokens / 1e9:.6f}B)"
    )
    print(
        f"chunks: {total_chunks:,}"
    )
    print(
        f"final bin size: "
        f"{total_tokens * 2 / (1024 ** 3):.2f} GiB"
    )

    en_files = sorted(
        (args.raw_root / "en").rglob(
            "*.parquet"
        )
    )

    zh_files = sorted(
        (args.raw_root / "zh").rglob(
            "*.parquet"
        )
    )

    code_files = sorted(
        (args.raw_root / "code").rglob(
            "*.json.gz"
        )
    )

    print()
    print(
        f"English raw files: {len(en_files)}"
    )
    print(
        f"Chinese raw files: {len(zh_files)}"
    )
    print(
        f"Code raw files:    {len(code_files)}"
    )

    if not en_files:
        raise FileNotFoundError(
            args.raw_root / "en"
        )

    if not zh_files:
        raise FileNotFoundError(
            args.raw_root / "zh"
        )

    if not code_files:
        raise FileNotFoundError(
            args.raw_root / "code"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    work_dir = (
        args.output_dir
        / f".{args.output_name}_work"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_path = (
        work_dir
        / "build_state.json"
    )

    work_paths = {
        source: (
            work_dir
            / f"{source}.uint16"
        )
        for source in SOURCE_ORDER
    }

    state = load_json(
        state_path
    )

    if state is None:
        state = create_initial_state(
            tokenizer_path=tokenizer_path,
            seq_len=args.seq_len,
            requested_targets=requested_targets,
            actual_targets=actual_targets,
            seed=args.seed,
        )

        save_json_atomic(
            state_path,
            state,
        )
    else:
        validate_state(
            state,
            args.seq_len,
            actual_targets,
        )

        print()
        print(
            f"[resume] state loaded from: {state_path}"
        )

    # ========================================================
    # English
    # ========================================================

    if not state["sources"]["en"]["completed"]:
        print()
        print(
            "=" * 72
        )
        print("BUILDING ENGLISH TOKENS")
        print("=" * 72)

        process_parquet_source(
            source="en",
            files=en_files,
            output_path=work_paths["en"],
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            target_tokens=actual_targets["en"],
            batch_docs=args.batch_docs,
            state=state,
            state_path=state_path,
        )
    else:
        print(
            "[en] already completed."
        )

    # ========================================================
    # Chinese
    # ========================================================

    if not state["sources"]["zh"]["completed"]:
        print()
        print(
            "=" * 72
        )
        print("BUILDING CHINESE TOKENS")
        print("=" * 72)

        process_parquet_source(
            source="zh",
            files=zh_files,
            output_path=work_paths["zh"],
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            target_tokens=actual_targets["zh"],
            batch_docs=args.batch_docs,
            state=state,
            state_path=state_path,
        )
    else:
        print(
            "[zh] already completed."
        )

    # ========================================================
    # Code
    # ========================================================

    if not state["sources"]["code"]["completed"]:
        print()
        print(
            "=" * 72
        )
        print("BUILDING CODE TOKENS")
        print("=" * 72)

        process_code_source(
            source="code",
            files=code_files,
            output_path=work_paths["code"],
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            target_tokens=actual_targets["code"],
            batch_docs=args.batch_docs,
            state=state,
            state_path=state_path,
        )
    else:
        print(
            "[code] already completed."
        )

    # ========================================================
    # Final mixed bin
    # ========================================================

    final_bin = (
        args.output_dir
        / f"{args.output_name}.bin"
    )

    final_meta = (
        args.output_dir
        / f"{args.output_name}.meta"
    )

    expected_final_bytes = (
        total_tokens * 2
    )

    final_is_valid = (
        final_bin.exists()
        and final_bin.stat().st_size
        == expected_final_bytes
        and final_meta.exists()
    )

    if not final_is_valid:
        build_final_bin(
            work_paths=work_paths,
            final_bin_path=final_bin,
            source_chunks=source_chunks,
            seq_len=args.seq_len,
            seed=args.seed,
        )

    source_meta = {}

    for source in SOURCE_ORDER:
        source_meta[source] = {
            "requested_tokens": requested_targets[source],
            "tokens": actual_targets[source],
            "chunks": source_chunks[source],
            "ratio": (
                actual_targets[source]
                / total_tokens
            ),
        }

    meta = {
        "version": VERSION,
        "dataset": args.output_name,
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "seq_len": args.seq_len,
        "num_chunks": total_chunks,
        "total_tokens": total_tokens,
        "dtype": "uint16",
        "shape": [
            total_chunks,
            args.seq_len,
        ],
        "tokenizer": str(
            tokenizer_path
        ),
        "mix_seed": args.seed,
        "sources": source_meta,
        "raw_root": str(
            args.raw_root
        ),
    }

    save_json_atomic(
        final_meta,
        meta,
    )

    print()
    print(
        "=" * 72
    )
    print("DONE")
    print("=" * 72)

    print(
        f"BIN:  {final_bin}"
    )
    print(
        f"META: {final_meta}"
    )
    print(
        f"Tokens: {total_tokens:,} "
        f"({total_tokens / 1e9:.6f}B)"
    )
    print(
        f"Chunks: {total_chunks:,}"
    )
    print(
        f"BIN size: "
        f"{final_bin.stat().st_size / (1024 ** 3):.2f} GiB"
    )

    print()
    print(
        "Source mixture:"
    )

    for source in SOURCE_ORDER:
        info = source_meta[source]

        print(
            f"  {source}: "
            f"{info['tokens'] / 1e9:.4f}B "
            f"({info['ratio'] * 100:.2f}%)"
        )


if __name__ == "__main__":
    main()