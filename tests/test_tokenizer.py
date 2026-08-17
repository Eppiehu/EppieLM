from pathlib import Path

from transformers import PreTrainedTokenizerFast


TOKENIZER_DIR = Path("tokenizer/eppielm_tokenizer")


def load_tokenizer() -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(
        TOKENIZER_DIR
    )


def test_tokenizer_files_exist():
    required_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]

    for filename in required_files:
        assert (TOKENIZER_DIR / filename).exists()


def test_tokenizer_vocab_size():
    tokenizer = load_tokenizer()

    assert len(tokenizer) == 15000


def test_special_token_ids_match_model_config():
    tokenizer = load_tokenizer()

    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 0


def test_chinese_round_trip():
    tokenizer = load_tokenizer()

    text = "北京大学人工智能研究正在快速发展。"

    input_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    decoded = tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
    )

    assert decoded == text


def test_english_round_trip():
    tokenizer = load_tokenizer()

    text = "EppieLM is a lightweight language model."

    input_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    decoded = tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
    )

    assert decoded == text


def test_code_round_trip():
    tokenizer = load_tokenizer()

    text = "def forward(x):\n    return self.proj(x)"

    input_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    decoded = tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
    )

    assert decoded == text


def test_token_ids_stay_inside_model_vocab():
    tokenizer = load_tokenizer()

    texts = [
        "北京大学人工智能",
        "EppieLM language model",
        "torch.nn.functional.scaled_dot_product_attention",
        "你好，世界！123456",
    ]

    for text in texts:
        input_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        assert input_ids

        assert max(input_ids) < 15000


def test_model_max_length():
    tokenizer = load_tokenizer()

    assert tokenizer.model_max_length == 8192


def test_chat_template():
    tokenizer = load_tokenizer()

    messages = [
        {
            "role": "user",
            "content": "你好。",
        },
        {
            "role": "assistant",
            "content": "你好！",
        },
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )

    assert "<|user|>" in rendered
    assert "<|assistant|>" in rendered
    assert "<|im_end|>" in rendered