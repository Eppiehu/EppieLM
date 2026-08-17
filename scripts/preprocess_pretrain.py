import argparse
import json
import multiprocessing as mp
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


_tokenizer = None
_eos_token_id = None
_vocab_size = None


STATUS_OK = 0
STATUS_INVALID_JSON = 1
STATUS_MISSING_TEXT = 2
STATUS_EMPTY_TEXT = 3
STATUS_TOKEN_ERROR = 4


def _init_worker(
    tokenizer_path: str,
) -> None:
    """
    每个 worker 独立加载 tokenizer。

    这样做避免 multiprocessing 跨进程共享 tokenizer，
    在 Windows 的 spawn 模式下也更稳定。
    """
    global _tokenizer
    global _eos_token_id
    global _vocab_size

    _tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )

    _eos_token_id = _tokenizer.eos_token_id
    _vocab_size = len(_tokenizer)

    if _eos_token_id is None:
        raise ValueError(
            "Tokenizer 没有 eos_token_id。"
        )


def _tokenize_line(
    line: str,
):
    """
    将单行 JSONL 转换成 token ids。

    输入格式：
        {"text": "..."}

    每篇文档末尾追加 EOS，
    保证连续 packing 后仍然存在文档边界。
    """
    try:
        line = line.strip()

        if not line:
            return (
                STATUS_EMPTY_TEXT,
                None,
            )

        try:
            item = json.loads(line)

        except json.JSONDecodeError:
            return (
                STATUS_INVALID_JSON,
                None,
            )

        if "text" not in item:
            return (
                STATUS_MISSING_TEXT,
                None,
            )

        text = item["text"]

        if not isinstance(
            text,
            str,
        ):
            return (
                STATUS_MISSING_TEXT,
                None,
            )

        text = text.strip()

        if not text:
            return (
                STATUS_EMPTY_TEXT,
                None,
            )

        token_ids = _tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        token_ids.append(
            _eos_token_id
        )

        if token_ids:
            max_token_id = max(
                token_ids
            )

            if max_token_id >= _vocab_size:
                raise ValueError(
                    "Tokenizer 产生了超出词表范围的 "
                    f"token id: {max_token_id}"
                )

        return (
            STATUS_OK,
            token_ids,
        )

    except Exception:
        return (
            STATUS_TOKEN_ERROR,
            None,
        )


def _line_iterator(
    input_path: Path,
) -> Iterator[str]:
    """
    utf-8-sig 同时兼容：

    - 普通 UTF-8
    - UTF-8 with BOM

    Windows PowerShell 生成的文本文件可能带 BOM。
    """
    with input_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        yield from f


def _count_lines(
    input_path: Path,
) -> int:
    with input_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        return sum(
            1
            for _ in f
        )


def preprocess(
    input_path: str,
    output_path: str,
    tokenizer_path: str,
    seq_len: int = 2048,
    num_workers: Optional[int] = None,
) -> None:
    input_path = Path(
        input_path
    )

    output_prefix = Path(
        output_path
    )

    tokenizer_path_obj = Path(
        tokenizer_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"输入数据不存在: {input_path}"
        )

    if not tokenizer_path_obj.exists():
        raise FileNotFoundError(
            "Tokenizer 目录不存在: "
            f"{tokenizer_path_obj}"
        )

    if seq_len <= 1:
        raise ValueError(
            "seq_len 必须大于 1。"
        )

    if num_workers is None:
        num_workers = min(
            8,
            os.cpu_count() or 1,
        )

    if num_workers < 1:
        raise ValueError(
            "num_workers 必须 >= 1。"
        )

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bin_path = Path(
        str(output_prefix)
        + ".bin"
    )

    meta_path = Path(
        str(output_prefix)
        + ".meta"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path_obj),
        local_files_only=True,
    )

    vocab_size = len(
        tokenizer
    )

    eos_token_id = tokenizer.eos_token_id

    if eos_token_id is None:
        raise ValueError(
            "Tokenizer 必须定义 eos_token_id。"
        )

    if vocab_size > 65536:
        raise ValueError(
            "当前二进制格式使用 uint16，"
            f"但 vocab_size={vocab_size} "
            "超过可表示范围。"
        )

    print("=" * 64)
    print(
        "EppieLM Pretraining Data Preprocessor"
    )
    print("=" * 64)

    print(
        f"Input:      {input_path}"
    )

    print(
        f"Output:     {bin_path}"
    )

    print(
        f"Tokenizer:  {tokenizer_path_obj}"
    )

    print(
        f"Vocab:      {vocab_size}"
    )

    print(
        f"EOS ID:     {eos_token_id}"
    )

    print(
        f"Seq length: {seq_len}"
    )

    print(
        f"Workers:    {num_workers}"
    )

    print("=" * 64)

    print(
        "\n[1/3] Counting documents..."
    )

    num_lines = _count_lines(
        input_path
    )

    print(
        f"Input lines: {num_lines:,}"
    )

    stats = {
        "input_lines": num_lines,
        "processed_documents": 0,
        "invalid_json": 0,
        "missing_text": 0,
        "empty_text": 0,
        "token_errors": 0,
    }

    total_tokens = 0
    buffer = []

    # 避免 Python list 无限增长。
    # 大约每 100 万 token 刷一次磁盘。
    buffer_limit = 1_000_000

    # 临时文件直接创建在最终输出目录。
    # 这样 Windows 下不会出现 C 盘 -> D 盘跨盘 replace。
    temp_file = tempfile.NamedTemporaryFile(
        mode="w+b",
        delete=False,
        suffix=".eppie.tmp",
        dir=str(
            output_prefix.parent
        ),
    )

    temp_path = Path(
        temp_file.name
    )

    def flush_buffer() -> None:
        nonlocal buffer

        if not buffer:
            return

        np.asarray(
            buffer,
            dtype=np.uint16,
        ).tofile(
            temp_file
        )

        buffer = []

    def consume_result(
        result,
    ) -> None:
        nonlocal total_tokens
        nonlocal buffer

        status, token_ids = result

        if status == STATUS_OK:
            stats[
                "processed_documents"
            ] += 1

            buffer.extend(
                token_ids
            )

            total_tokens += len(
                token_ids
            )

        elif status == STATUS_INVALID_JSON:
            stats[
                "invalid_json"
            ] += 1

        elif status == STATUS_MISSING_TEXT:
            stats[
                "missing_text"
            ] += 1

        elif status == STATUS_EMPTY_TEXT:
            stats[
                "empty_text"
            ] += 1

        else:
            stats[
                "token_errors"
            ] += 1

        if len(buffer) >= buffer_limit:
            flush_buffer()

    try:
        print(
            "\n[2/3] Tokenizing..."
        )

        if num_workers == 1:
            # 单进程适合 smoke test 和调试。
            _init_worker(
                str(
                    tokenizer_path_obj
                )
            )

            iterator = map(
                _tokenize_line,
                _line_iterator(
                    input_path
                ),
            )

            for result in tqdm(
                iterator,
                total=num_lines,
            ):
                consume_result(
                    result
                )

        else:
            # Windows 使用 spawn；
            # Linux 下 fork 启动成本更低。
            ctx = mp.get_context(
                "spawn"
                if os.name == "nt"
                else "fork"
            )

            with ctx.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(
                    str(
                        tokenizer_path_obj
                    ),
                ),
            ) as pool:
                iterator = pool.imap(
                    _tokenize_line,
                    _line_iterator(
                        input_path
                    ),
                    chunksize=128,
                )

                for result in tqdm(
                    iterator,
                    total=num_lines,
                ):
                    consume_result(
                        result
                    )

        flush_buffer()

        temp_file.flush()
        temp_file.close()

        num_chunks = (
            total_tokens
            // seq_len
        )

        kept_tokens = (
            num_chunks
            * seq_len
        )

        dropped_tokens = (
            total_tokens
            - kept_tokens
        )

        if num_chunks == 0:
            raise ValueError(
                "有效 token 数不足一个完整 "
                f"{seq_len}-token chunk。"
            )

        # 只保留完整 chunk。
        #
        # 直接 truncate 二进制文件，
        # 避免重新把所有 token 加载进内存。
        with temp_path.open(
            "r+b"
        ) as f:
            f.truncate(
                kept_tokens
                * np.dtype(
                    np.uint16
                ).itemsize
            )

        if bin_path.exists():
            bin_path.unlink()

        # 临时文件目前已经在相同目录，
        # shutil.move 同时还能兼容未来可能的跨盘情况。
        shutil.move(
            str(temp_path),
            str(bin_path),
        )

        print(
            "\n[3/3] Writing metadata..."
        )

        meta = {
            "format_version": 1,
            "dtype": "uint16",
            "vocab_size": vocab_size,
            "eos_token_id": eos_token_id,
            "seq_len": seq_len,
            "num_chunks": num_chunks,
            "shape": [
                num_chunks,
                seq_len,
            ],
            "total_tokens_before_truncation": (
                total_tokens
            ),
            "kept_tokens": kept_tokens,
            "dropped_tokens": dropped_tokens,
            "documents": stats,
        }

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

        size_mb = (
            bin_path.stat().st_size
            / 1024
            / 1024
        )

        drop_ratio = (
            dropped_tokens
            / total_tokens
            if total_tokens
            else 0.0
        )

        print()
        print("=" * 64)
        print(
            "Preprocessing finished"
        )
        print("=" * 64)

        print(
            f"Processed documents: "
            f"{stats['processed_documents']:,}"
        )

        print(
            f"Invalid JSON:        "
            f"{stats['invalid_json']:,}"
        )

        print(
            f"Missing text:        "
            f"{stats['missing_text']:,}"
        )

        print(
            f"Empty text:          "
            f"{stats['empty_text']:,}"
        )

        print(
            f"Token errors:        "
            f"{stats['token_errors']:,}"
        )

        print(
            f"Total tokens:        "
            f"{total_tokens:,}"
        )

        print(
            f"Kept tokens:         "
            f"{kept_tokens:,}"
        )

        print(
            f"Chunks:              "
            f"{num_chunks:,}"
        )

        print(
            f"Dropped tokens:      "
            f"{dropped_tokens:,} "
            f"({drop_ratio:.4%})"
        )

        print(
            f"Binary size:         "
            f"{size_mb:.2f} MB"
        )

        print()

        print(
            f"BIN:  {bin_path}"
        )

        print(
            f"META: {meta_path}"
        )

    finally:
        try:
            temp_file.close()
        except Exception:
            pass

        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "EppieLM 预训练数据预处理："
            "JSONL -> token stream -> uint16 .bin"
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            'JSONL 文件，每行格式为 '
            '{"text": "..."}'
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "输出前缀，例如 "
            "data/pretrain/train"
        ),
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=(
            "tokenizer/"
            "eppielm_tokenizer"
        ),
        help="Tokenizer 路径",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="每个预训练 chunk 的 token 数",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help=(
            "Tokenizer worker 数量；"
            "默认最多使用 8 个"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    preprocess(
        input_path=args.input,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    # Windows multiprocessing 需要 main guard。
    mp.freeze_support()

    main()