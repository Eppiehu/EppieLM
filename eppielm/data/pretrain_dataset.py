import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class EppiePretrainDataset(Dataset):
    """
    EppieLM 预训练数据集。

    数据由 preprocess_pretrain.py 提前处理成：
        train.bin
        train.meta

    .bin 使用连续 uint16 保存 token，
    .meta 保存 shape、seq_len、vocab_size 等信息。

    Dataset 通过 np.memmap 访问数据，
    即使训练集很大也不需要一次加载进内存。
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        seq_len: Optional[int] = None,
        expected_vocab_size: Optional[int] = None,
    ):
        data_path = Path(data_path)

        if data_path.suffix != ".bin":
            data_path = data_path.with_suffix(".bin")

        if not data_path.exists():
            raise FileNotFoundError(
                f"预训练数据不存在: {data_path}"
            )

        meta_path = data_path.with_suffix(".meta")

        if not meta_path.exists():
            raise FileNotFoundError(
                f"预训练数据 meta 不存在: {meta_path}"
            )

        with meta_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.meta = json.load(f)

        self.data_path = data_path
        self.meta_path = meta_path

        self._validate_meta(
            seq_len=seq_len,
            expected_vocab_size=expected_vocab_size,
        )

        self.seq_len = int(
            self.meta["seq_len"]
        )

        self.num_chunks = int(
            self.meta["num_chunks"]
        )

        self.vocab_size = int(
            self.meta["vocab_size"]
        )

        self.dtype = np.dtype(
            self.meta["dtype"]
        )

        self.shape = tuple(
            self.meta["shape"]
        )

        # memmap 不会把整个 .bin 文件读进内存。
        # 对几十 GB 甚至更大的预训练数据尤其重要。
        self.data = np.memmap(
            self.data_path,
            dtype=self.dtype,
            mode="r",
            shape=self.shape,
        )

    def _validate_meta(
        self,
        seq_len: Optional[int],
        expected_vocab_size: Optional[int],
    ) -> None:
        required_fields = {
            "vocab_size",
            "seq_len",
            "num_chunks",
            "dtype",
            "shape",
        }

        missing = (
            required_fields
            - set(self.meta.keys())
        )

        if missing:
            raise ValueError(
                "meta 缺少字段: "
                + ", ".join(sorted(missing))
            )

        if self.meta["dtype"] != "uint16":
            raise ValueError(
                "EppieLM 当前预训练数据要求 "
                f"uint16，实际为 {self.meta['dtype']}。"
            )

        meta_seq_len = int(
            self.meta["seq_len"]
        )

        meta_num_chunks = int(
            self.meta["num_chunks"]
        )

        shape = self.meta["shape"]

        if (
            not isinstance(shape, list)
            or len(shape) != 2
        ):
            raise ValueError(
                f"非法 shape: {shape}"
            )

        expected_shape = [
            meta_num_chunks,
            meta_seq_len,
        ]

        if shape != expected_shape:
            raise ValueError(
                "meta shape 与 num_chunks / seq_len "
                f"不一致: shape={shape}, "
                f"expected={expected_shape}"
            )

        if (
            seq_len is not None
            and meta_seq_len != seq_len
        ):
            raise ValueError(
                "seq_len mismatch: "
                f"data={meta_seq_len}, "
                f"requested={seq_len}"
            )

        meta_vocab_size = int(
            self.meta["vocab_size"]
        )

        if (
            expected_vocab_size is not None
            and meta_vocab_size
            > expected_vocab_size
        ):
            raise ValueError(
                "数据词表大小超过模型词表大小: "
                f"data={meta_vocab_size}, "
                f"model={expected_vocab_size}"
            )

        # uint16 可以表示 0~65535。
        # 当前 EppieLM 15K tokenizer 完全适合这种存储格式。
        if meta_vocab_size > 65536:
            raise ValueError(
                "vocab_size 超过 uint16 可表示范围。"
            )

        expected_bytes = (
            meta_num_chunks
            * meta_seq_len
            * np.dtype(np.uint16).itemsize
        )

        actual_bytes = (
            self.data_path.stat().st_size
        )

        if actual_bytes != expected_bytes:
            raise ValueError(
                "bin 文件大小与 meta 不匹配: "
                f"expected={expected_bytes} bytes, "
                f"actual={actual_bytes} bytes"
            )

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(
        self,
        idx: int,
    ):
        if idx < 0:
            idx += self.num_chunks

        if (
            idx < 0
            or idx >= self.num_chunks
        ):
            raise IndexError(
                f"index {idx} out of range "
                f"for dataset of size "
                f"{self.num_chunks}"
            )

        # memmap 的 uint16 转成 PyTorch 训练需要的 int64。
        chunk = np.asarray(
            self.data[idx],
            dtype=np.int64,
        )

        input_ids = torch.from_numpy(
            chunk.copy()
        )

        # CausalLM 内部负责 shift：
        #
        # logits[..., :-1]
        # labels[..., 1:]
        #
        # 因此这里 input_ids 与 labels 保持相同即可。
        labels = input_ids.clone()

        return input_ids, labels

    def __repr__(self) -> str:
        return (
            "EppiePretrainDataset("
            f"chunks={self.num_chunks:,}, "
            f"seq_len={self.seq_len}, "
            f"vocab_size={self.vocab_size}, "
            f"dtype={self.dtype}"
            ")"
        )