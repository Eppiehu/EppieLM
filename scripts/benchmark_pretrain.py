import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eppielm.models.configuration_eppielm import EppieLMConfig
from eppielm.models.modeling_eppielm import EppieLMForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark EppieLM pretraining."
    )

    parser.add_argument(
        "--bin",
        type=Path,
        default=Path(
            r"E:\EppieLMData\pretrain\eppielm_3b.bin"
        ),
    )

    parser.add_argument(
        "--meta",
        type=Path,
        default=Path(
            r"E:\EppieLMData\pretrain\eppielm_3b.meta"
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help=(
            "Checkpoint every N-th transformer layer. "
            "1 means every layer."
        ),
    )

    parser.add_argument(
        "--sdpa-backend",
        type=str,
        default="cudnn",
        choices=[
            "auto",
            "efficient",
            "cudnn",
            "math",
            "flash",
        ],
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the model.",
    )

    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
        ],
        help="torch.compile mode.",
    )

    return parser.parse_args()


def load_meta(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"META not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def create_model(
    seq_len: int,
):
    config = EppieLMConfig(
        vocab_size=15000,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=22,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        initializer_range=0.02,
        attention_impl="sdpa",
        use_cache=False,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )

    if seq_len > config.max_position_embeddings:
        raise ValueError(
            f"seq_len={seq_len} exceeds "
            f"max_position_embeddings="
            f"{config.max_position_embeddings}"
        )

    return EppieLMForCausalLM(
        config
    )


def create_optimizer(
    model,
    lr: float,
):
    kwargs = {
        "lr": lr,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            fused=True,
            **kwargs,
        )

        fused = True

    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            **kwargs,
        )

        fused = False

    return optimizer, fused


def format_gib(
    num_bytes: int,
) -> str:
    return (
        f"{num_bytes / (1024 ** 3):.2f} GiB"
    )


def get_sdpa_context(
    backend_name: str,
):
    if backend_name == "auto":
        return nullcontext()

    backend_map = {
        "efficient": (
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
        ),
        "cudnn": (
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION
        ),
        "math": (
            torch.nn.attention.SDPBackend.MATH
        ),
        "flash": (
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ),
    }

    return torch.nn.attention.sdpa_kernel(
        backend_map[backend_name]
    )


def main():
    args = parse_args()

    if args.checkpoint_every < 1:
        raise ValueError(
            "--checkpoint-every must be >= 1"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    torch.manual_seed(
        args.seed
    )

    torch.cuda.manual_seed_all(
        args.seed
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    device = torch.device(
        "cuda"
    )

    props = torch.cuda.get_device_properties(
        device
    )

    print("=" * 72)
    print("EppieLM PRETRAIN BENCHMARK")
    print("=" * 72)

    print(
        f"GPU: {props.name}"
    )

    print(
        f"VRAM: "
        f"{format_gib(props.total_memory)}"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )

    print(
        f"SDPA backend: "
        f"{args.sdpa_backend}"
    )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 is not supported."
        )

    meta = load_meta(
        args.meta
    )

    seq_len = int(
        meta["seq_len"]
    )

    num_chunks = int(
        meta["num_chunks"]
    )

    if meta["dtype"] != "uint16":
        raise RuntimeError(
            "Dataset must use uint16."
        )

    shape = (
        num_chunks,
        seq_len,
    )

    expected_tokens = (
        num_chunks
        * seq_len
    )

    expected_bytes = (
        expected_tokens
        * 2
    )

    if not args.bin.exists():
        raise FileNotFoundError(
            f"BIN not found: {args.bin}"
        )

    actual_bytes = (
        args.bin.stat().st_size
    )

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"BIN size mismatch: "
            f"{actual_bytes:,} != "
            f"{expected_bytes:,}"
        )

    print()
    print("Dataset")
    print("-" * 72)

    print(
        f"BIN: {args.bin}"
    )

    print(
        f"Shape: {shape}"
    )

    print(
        f"Tokens: {expected_tokens:,}"
    )

    print(
        f"Size: "
        f"{format_gib(actual_bytes)}"
    )

    data = np.memmap(
        args.bin,
        mode="r",
        dtype=np.uint16,
        shape=shape,
    )

    print()
    print("Building model...")

    model = create_model(
        seq_len
    )

    param_count = sum(
        p.numel()
        for p in model.parameters()
    )

    model.to(
        device
    )

    model.train()

    # 保留原始模型引用。
    # optimizer / gradient clipping / 模型结构统计都使用 raw_model。
    raw_model = model

    gradient_checkpointing = (
        not args.no_gradient_checkpointing
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

        model.set_checkpoint_every(
            args.checkpoint_every
        )

    optimizer, fused = create_optimizer(
        raw_model,
        args.lr,
    )

    if args.compile:
        print()
        print(
            f"Compiling model "
            f"(mode={args.compile_mode})..."
        )

        model = torch.compile(
            raw_model,
            mode=args.compile_mode,
            fullgraph=False,
            dynamic=False,
        )

    else:
        model = raw_model

    effective_batch = (
        args.micro_batch_size
        * args.grad_accum
    )

    tokens_per_step = (
        effective_batch
        * seq_len
    )

    checkpointed_layers = (
        raw_model.model.get_checkpointed_layer_count()
        if gradient_checkpointing
        else 0
    )

    total_layers = len(
        raw_model.model.layers
    )

    print()
    print("Model")
    print("-" * 72)

    print(
        f"Parameters: "
        f"{param_count / 1e6:.2f}M"
    )

    print(
        f"Sequence length: "
        f"{seq_len}"
    )

    print(
        f"Micro batch size: "
        f"{args.micro_batch_size}"
    )

    print(
        f"Gradient accumulation: "
        f"{args.grad_accum}"
    )

    print(
        f"Effective batch size: "
        f"{effective_batch}"
    )

    print(
        f"Tokens / optimizer step: "
        f"{tokens_per_step:,}"
    )

    print(
        f"Gradient checkpointing: "
        f"{gradient_checkpointing}"
    )

    if gradient_checkpointing:
        print(
            f"Checkpoint every: "
            f"{args.checkpoint_every}"
        )

        print(
            f"Checkpointed layers: "
            f"{checkpointed_layers}/"
            f"{total_layers}"
        )

        print(
            "Preserve RNG state: "
            f"{raw_model.model.checkpoint_preserve_rng_state}"
        )

    print(
        f"Optimizer fused: {fused}"
    )

    print(
        f"torch.compile: {args.compile}"
    )

    if args.compile:
        print(
            f"Compile mode: "
            f"{args.compile_mode}"
        )

    print(
        "Precision: BF16"
    )

    print(
        f"SDPA backend: "
        f"{args.sdpa_backend}"
    )

    warmup_steps = (
        max(args.warmup_steps, 10)
        if args.compile
        else args.warmup_steps
    )

    total_steps = (
        warmup_steps
        + args.steps
    )

    required_samples = (
        total_steps
        * args.grad_accum
        * args.micro_batch_size
    )

    if required_samples > num_chunks:
        raise RuntimeError(
            "Benchmark requires more samples "
            "than available in dataset."
        )

    rng = np.random.default_rng(
        args.seed
    )

    sample_indices = rng.integers(
        low=0,
        high=num_chunks,
        size=required_samples,
        dtype=np.int64,
    )

    cursor = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad(
        set_to_none=True
    )

    measured_times = []
    measured_losses = []

    print()
    print("=" * 72)

    print(
        f"WARMUP {warmup_steps} STEPS "
        f"+ BENCHMARK {args.steps} STEPS"
    )

    print("=" * 72)

    with get_sdpa_context(
        args.sdpa_backend
    ):
        for step in range(
            total_steps
        ):
            is_warmup = (
                step < warmup_steps
            )

            # compile 首次编译和 warmup 的显存峰值
            # 不计入正式 benchmark。
            if step == warmup_steps:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            torch.cuda.synchronize()

            start_time = (
                time.perf_counter()
            )

            accumulated_loss = 0.0

            for _ in range(
                args.grad_accum
            ):
                indices = sample_indices[
                    cursor:
                    cursor
                    + args.micro_batch_size
                ]

                cursor += (
                    args.micro_batch_size
                )

                batch_np = np.asarray(
                    data[indices],
                    dtype=np.int64,
                ).copy()

                input_ids = (
                    torch.from_numpy(
                        batch_np
                    )
                    .pin_memory()
                    .to(
                        device,
                        non_blocking=True,
                    )
                )

                labels = (
                    input_ids.clone()
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    outputs = model(
                        input_ids=input_ids,
                        labels=labels,
                        use_cache=False,
                    )

                    loss = outputs.loss

                    scaled_loss = (
                        loss
                        / args.grad_accum
                    )

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "Non-finite loss: "
                        f"{loss.item()}"
                    )

                scaled_loss.backward()

                accumulated_loss += (
                    loss.detach()
                    .float()
                    .item()
                )

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(),
                    max_norm=1.0,
                )
            )

            if not torch.isfinite(
                grad_norm
            ):
                raise RuntimeError(
                    "Non-finite gradient norm: "
                    f"{grad_norm}"
                )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            torch.cuda.synchronize()

            step_time = (
                time.perf_counter()
                - start_time
            )

            avg_loss = (
                accumulated_loss
                / args.grad_accum
            )

            throughput = (
                tokens_per_step
                / step_time
            )

            memory = (
                torch.cuda.memory_allocated()
            )

            peak_memory = (
                torch.cuda.max_memory_allocated()
            )

            phase = (
                "warmup"
                if is_warmup
                else "bench"
            )

            if not is_warmup:
                measured_times.append(
                    step_time
                )

                measured_losses.append(
                    avg_loss
                )

            print(
                f"[{phase}] "
                f"step {step + 1:>3}/"
                f"{total_steps} | "
                f"loss={avg_loss:.4f} | "
                f"time={step_time:.3f}s | "
                f"{throughput:,.0f} tok/s | "
                f"grad={float(grad_norm):.3f} | "
                f"mem={format_gib(memory)} | "
                f"peak={format_gib(peak_memory)}"
            )

    total_measured_time = sum(
        measured_times
    )

    measured_tokens = (
        args.steps
        * tokens_per_step
    )

    average_throughput = (
        measured_tokens
        / total_measured_time
    )

    average_step_time = (
        total_measured_time
        / args.steps
    )

    average_loss = (
        sum(measured_losses)
        / len(measured_losses)
    )

    peak_memory = (
        torch.cuda.max_memory_allocated()
    )

    estimated_seconds = (
        3_000_000_000
        / average_throughput
    )

    estimated_hours = (
        estimated_seconds
        / 3600
    )

    estimated_days = (
        estimated_hours
        / 24
    )

    print()
    print("=" * 72)
    print("BENCHMARK RESULT")
    print("=" * 72)

    print(
        f"SDPA backend: "
        f"{args.sdpa_backend}"
    )

    print(
        f"Checkpoint every: "
        f"{args.checkpoint_every}"
    )

    print(
        f"Checkpointed layers: "
        f"{checkpointed_layers}/"
        f"{total_layers}"
    )

    print(
        f"Measured steps: "
        f"{args.steps}"
    )

    print(
        f"Measured tokens: "
        f"{measured_tokens:,}"
    )

    print(
        f"Average step time: "
        f"{average_step_time:.3f} s"
    )

    print(
        f"Average throughput: "
        f"{average_throughput:,.0f} "
        f"tokens/s"
    )

    print(
        f"Average loss: "
        f"{average_loss:.4f}"
    )

    print(
        f"Peak allocated VRAM: "
        f"{format_gib(peak_memory)}"
    )

    print()
    print(
        "Estimated 3B training time"
    )

    print("-" * 72)

    print(
        f"{estimated_hours:.1f} hours"
    )

    print(
        f"{estimated_days:.2f} days"
    )


if __name__ == "__main__":
    main()