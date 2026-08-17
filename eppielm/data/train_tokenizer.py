import argparse
import json
from pathlib import Path
from typing import Iterator, List

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<think>",
    "</think>",
]


def iter_text_files(
    paths: List[Path],
) -> Iterator[str]:
    """
    按行读取纯文本文件。

    tokenizer 训练不需要一次把整个语料加载进内存，
    这里使用生成器是为了以后能直接处理更大的预训练语料。
    """
    for path in paths:
        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:
            for line in file:
                text = line.strip()

                if text:
                    yield text


def iter_jsonl_files(
    paths: List[Path],
    text_field: str,
) -> Iterator[str]:
    """
    从 JSONL 中提取指定文本字段。

    第一版只接受明确的文本字段，不自动猜字段名称。
    数据格式越明确，后续训练数据流水线越容易复现。
    """
    for path in paths:
        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON 解析失败: {path}:{line_number}"
                    ) from exc

                text = item.get(text_field)

                if text is None:
                    continue

                if not isinstance(text, str):
                    continue

                text = text.strip()

                if text:
                    yield text


def collect_input_files(
    input_paths: List[str],
) -> List[Path]:
    """
    支持同时输入文件和目录。

    目录下自动收集 .txt 和 .jsonl，
    方便后续把不同来源的 tokenizer 训练语料放在同一目录。
    """
    files = []

    for raw_path in input_paths:
        path = Path(raw_path)

        if not path.exists():
            raise FileNotFoundError(
                f"输入路径不存在: {path}"
            )

        if path.is_file():
            files.append(path)
            continue

        for suffix in ("*.txt", "*.jsonl"):
            files.extend(
                sorted(path.rglob(suffix))
            )

    if not files:
        raise ValueError(
            "没有找到可用于 tokenizer 训练的 .txt 或 .jsonl 文件。"
        )

    return files


def build_text_iterator(
    files: List[Path],
    text_field: str,
) -> Iterator[str]:
    txt_files = [
        path
        for path in files
        if path.suffix.lower() == ".txt"
    ]

    jsonl_files = [
        path
        for path in files
        if path.suffix.lower() == ".jsonl"
    ]

    if txt_files:
        yield from iter_text_files(
            txt_files
        )

    if jsonl_files:
        yield from iter_jsonl_files(
            jsonl_files,
            text_field=text_field,
        )


def train_tokenizer(
    input_paths: List[str],
    output_dir: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    text_field: str = "text",
) -> None:
    files = collect_input_files(
        input_paths
    )

    output_path = Path(
        output_dir
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = Tokenizer(
        BPE(
            unk_token="<|unk|>",
        )
    )

    # NFKC 可以减少全角字符、兼容字符等造成的不必要词表浪费。
    # 这对中英文混合语料尤其有用。
    tokenizer.normalizer = NFKC()

    # ByteLevel 可以保证任意 UTF-8 文本都能够被编码，
    # 不需要为中文、英文、代码分别维护不同的预切分逻辑。
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    text_iterator = build_text_iterator(
        files,
        text_field=text_field,
    )

    tokenizer.train_from_iterator(
        text_iterator,
        trainer=trainer,
    )

    bos_id = tokenizer.token_to_id(
        "<|bos|>"
    )

    eos_id = tokenizer.token_to_id(
        "<|eos|>"
    )

    # 普通预训练文本默认采用：
    #
    # <|bos|> text <|eos|>
    #
    # Chat 格式以后由 chat_template 单独处理，
    # 不把两种格式硬编码在同一个训练步骤里。
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A $B <|eos|>",
        special_tokens=[
            ("<|bos|>", bos_id),
            ("<|eos|>", eos_id),
        ],
    )

    tokenizer_json_path = (
        output_path
        / "tokenizer.json"
    )

    tokenizer.save(
        str(tokenizer_json_path)
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        additional_special_tokens=[
            "<|im_start|>",
            "<|im_end|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<think>",
            "</think>",
        ],
        model_max_length=8192,
        clean_up_tokenization_spaces=False,
    )

    # 第一版 chat template 保持简单。
    # system/user/assistant 的边界都显式写进 token 流，
    # 后面做 SFT loss mask 时更容易确定哪些 token 应该计算 loss。
    fast_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' }}"
        "{{ '<|' + message['role'] + '|>' }}"
        "{{ '\\n' }}"
        "{{ message['content'] }}"
        "{{ '<|im_end|>' }}"
        "{{ '\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|><|assistant|>\\n' }}"
        "{% endif %}"
    )

    fast_tokenizer.save_pretrained(
        output_path
    )

    print()
    print("EppieLM tokenizer 训练完成")
    print(
        f"输入文件数量: {len(files)}"
    )
    print(
        f"目标词表大小: {vocab_size}"
    )
    print(
        f"实际词表大小: {len(fast_tokenizer)}"
    )
    print(
        f"保存目录: {output_path.resolve()}"
    )

    print()
    print("特殊 Token:")

    for token in SPECIAL_TOKENS:
        print(
            f"{token:20s} -> "
            f"{fast_tokenizer.convert_tokens_to_ids(token)}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "训练 EppieLM ByteLevel BPE tokenizer"
        )
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help=(
            "训练语料文件或目录，"
            "支持 .txt 和 .jsonl"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="tokenizer/eppielm_tokenizer",
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
    )

    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help=(
            "JSONL 中保存文本的字段名"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_tokenizer(
        input_paths=args.input,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        text_field=args.text_field,
    )


if __name__ == "__main__":
    main()