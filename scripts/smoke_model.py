
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from eppielm import (
    EppieLMConfig,
    EppieLMForCausalLM,
)


TOKENIZER_DIR = Path(
    "tokenizer/eppielm_tokenizer"
)


def main():
    tokenizer = (
        PreTrainedTokenizerFast.from_pretrained(
            TOKENIZER_DIR
        )
    )

    config = EppieLMConfig(
        attention_impl="sdpa",
        use_cache=True,
    )

    model = EppieLMForCausalLM(
        config
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    text = "北京大学人工智能"

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encoded[
        "input_ids"
    ].to(device)

    attention_mask = encoded[
        "attention_mask"
    ].to(device)

    print("=" * 60)
    print("EppieLM-150M Smoke Test")
    print("=" * 60)

    print(
        f"Device: {device}"
    )

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    print(
        f"Tokenizer vocab: {len(tokenizer)}"
    )

    print(
        f"Model vocab: {config.vocab_size}"
    )

    print()
    print(
        f"Input text: {text}"
    )

    print(
        f"Input IDs: {input_ids.tolist()}"
    )

    print(
        f"Sequence length: {input_ids.shape[1]}"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

    print()
    print(
        f"Logits shape: "
        f"{tuple(outputs.logits.shape)}"
    )

    print(
        f"Logits finite: "
        f"{torch.isfinite(outputs.logits).all().item()}"
    )

    print(
        f"KV cache layers: "
        f"{len(outputs.past_key_values)}"
    )

    next_token_id = (
        outputs.logits[
            0,
            -1,
        ]
        .argmax()
        .item()
    )

    next_token = tokenizer.decode(
        [next_token_id]
    )

    print()
    print(
        f"Greedy next token ID: "
        f"{next_token_id}"
    )

    print(
        f"Greedy next token: "
        f"{repr(next_token)}"
    )

    labels = input_ids.clone()

    with torch.no_grad():
        loss_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

    print()
    print(
        f"Random-init LM loss: "
        f"{loss_outputs.loss.item():.4f}"
    )

    print()
    print(
        "Smoke test passed."
    )


if __name__ == "__main__":
    main()