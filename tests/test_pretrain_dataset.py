import json
from pathlib import Path

import numpy as np
import pytest
import torch

from eppielm.data import (
    EppiePretrainDataset,
)


def create_dummy_dataset(
    tmp_path: Path,
    num_chunks: int = 4,
    seq_len: int = 16,
    vocab_size: int = 15000,
):
    bin_path = (
        tmp_path
        / "train.bin"
    )

    meta_path = (
        tmp_path
        / "train.meta"
    )

    data = np.arange(
        num_chunks * seq_len,
        dtype=np.uint16,
    )

    data %= min(
        vocab_size,
        1000,
    )

    data = data.reshape(
        num_chunks,
        seq_len,
    )

    data.tofile(
        bin_path
    )

    meta = {
        "format_version": 1,
        "dtype": "uint16",
        "vocab_size": vocab_size,
        "eos_token_id": 2,
        "seq_len": seq_len,
        "num_chunks": num_chunks,
        "shape": [
            num_chunks,
            seq_len,
        ],
        "total_tokens_before_truncation": (
            num_chunks
            * seq_len
        ),
        "kept_tokens": (
            num_chunks
            * seq_len
        ),
        "dropped_tokens": 0,
        "documents": {
            "input_lines": 4,
            "processed_documents": 4,
            "invalid_json": 0,
            "missing_text": 0,
            "empty_text": 0,
            "token_errors": 0,
        },
    }

    with meta_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            meta,
            f,
        )

    return bin_path


def test_dataset_length(
    tmp_path,
):
    bin_path = (
        create_dummy_dataset(
            tmp_path,
            num_chunks=4,
            seq_len=16,
        )
    )

    dataset = (
        EppiePretrainDataset(
            bin_path,
            seq_len=16,
            expected_vocab_size=15000,
        )
    )

    assert len(dataset) == 4


def test_dataset_sample_shape(
    tmp_path,
):
    bin_path = (
        create_dummy_dataset(
            tmp_path,
            num_chunks=4,
            seq_len=16,
        )
    )

    dataset = (
        EppiePretrainDataset(
            bin_path,
            seq_len=16,
            expected_vocab_size=15000,
        )
    )

    input_ids, labels = (
        dataset[0]
    )

    assert input_ids.shape == (
        16,
    )

    assert labels.shape == (
        16,
    )

    assert (
        input_ids.dtype
        == torch.long
    )

    assert (
        labels.dtype
        == torch.long
    )


def test_labels_match_input_ids(
    tmp_path,
):
    bin_path = (
        create_dummy_dataset(
            tmp_path
        )
    )

    dataset = (
        EppiePretrainDataset(
            bin_path
        )
    )

    input_ids, labels = (
        dataset[0]
    )

    assert torch.equal(
        input_ids,
        labels,
    )

    # 两个 tensor 不应该共享同一块 storage，
    # 防止后续修改 labels 意外改变 input_ids。
    labels[0] = -100

    assert (
        input_ids[0]
        != labels[0]
    )


def test_seq_len_mismatch(
    tmp_path,
):
    bin_path = (
        create_dummy_dataset(
            tmp_path,
            seq_len=16,
        )
    )

    with pytest.raises(
        ValueError,
        match="seq_len mismatch",
    ):
        EppiePretrainDataset(
            bin_path,
            seq_len=32,
        )


def test_vocab_size_validation(
    tmp_path,
):
    bin_path = (
        create_dummy_dataset(
            tmp_path,
            vocab_size=15000,
        )
    )

    with pytest.raises(
        ValueError,
        match="词表大小",
    ):
        EppiePretrainDataset(
            bin_path,
            expected_vocab_size=10000,
        )


def test_bin_size_validation(
    tmp_path,
):
    bin_path = (
        create_dummy_dataset(
            tmp_path,
            num_chunks=4,
            seq_len=16,
        )
    )

    # 人为破坏二进制文件，
    # 验证 Dataset 能在训练开始前发现损坏。
    with bin_path.open(
        "ab"
    ) as f:
        f.write(
            b"\x00\x00"
        )

    with pytest.raises(
        ValueError,
        match="bin 文件大小",
    ):
        EppiePretrainDataset(
            bin_path
        )


def test_negative_index(
    tmp_path,
):
    bin_path = (
        create_dummy_dataset(
            tmp_path,
            num_chunks=4,
        )
    )

    dataset = (
        EppiePretrainDataset(
            bin_path
        )
    )

    first = dataset[0][0]
    last = dataset[-1][0]

    assert not torch.equal(
        first,
        last,
    )


def test_index_out_of_range(
    tmp_path,
):
    bin_path = (
        create_dummy_dataset(
            tmp_path,
            num_chunks=4,
        )
    )

    dataset = (
        EppiePretrainDataset(
            bin_path
        )
    )

    with pytest.raises(
        IndexError
    ):
        _ = dataset[4]