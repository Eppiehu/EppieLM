import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eppielm.models import EppieLMConfig, EppieLMForCausalLM


MODEL_DIR = Path(
    "/root/autodl-tmp/releases/EppieLM-150M-Base"
)
TOKENIZER_DIR = (
    MODEL_DIR / "tokenizer" / "eppielm_tokenizer"
)


def load_model(device: torch.device):
    with (MODEL_DIR / "config.json").open(
        "r",
        encoding="utf-8",
    ) as f:
        config_dict = json.load(f)

    config = EppieLMConfig.from_dict(
        config_dict
    )

    model = EppieLMForCausalLM(config)

    state_dict = torch.load(
        MODEL_DIR / "model.pt",
        map_location="cpu",
        weights_only=True,
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        print("Missing keys:", missing)

    if unexpected:
        print("Unexpected keys:", unexpected)

    model = model.to(
        device=device,
        dtype=(
            torch.bfloat16
            if device.type == "cuda"
            else torch.float32
        ),
    )

    model.eval()

    return model


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
    do_sample: bool = False,
):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encoded["input_ids"].to(
        device
    )

    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
        )

        logits = outputs.logits[:, -1, :]

        if do_sample:
            logits = logits / temperature

            if top_k > 0:
                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1)),
                )

                cutoff = values[:, -1].unsqueeze(-1)

                logits = logits.masked_fill(
                    logits < cutoff,
                    float("-inf"),
                )

            probs = torch.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

        else:
            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True,
            )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=-1,
        )

        if (
            eos_token_id is not None
            and next_token.item()
            == eos_token_id
        ):
            break

    return tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
    )


def main():
    random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("EppieLM-150M-Base Generation Test")
    print("=" * 72)
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        local_files_only=True,
        use_fast=True,
    )

    model = load_model(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {num_params / 1e6:.2f}M"
    )
    print(
        f"Tokenizer vocab: {len(tokenizer):,}"
    )
    print()

    prompts = [
        "中国的首都是",
        "北京大学是一所",
        "人工智能的发展",
        "The future of artificial intelligence is",
        "Machine learning is",
        "def fibonacci(n):",
        "import torch",
    ]

    for prompt in prompts:
        print("=" * 72)
        print(f"PROMPT: {prompt}")

        greedy = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=64,
            do_sample=False,
        )

        print("\n[GREEDY]")
        print(greedy)

        sampled = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=64,
            temperature=0.8,
            top_k=50,
            do_sample=True,
        )

        print("\n[SAMPLE temperature=0.8 top_k=50]")
        print(sampled)
        print()


if __name__ == "__main__":
    main()
