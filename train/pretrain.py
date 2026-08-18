import argparse
import json
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


from eppielm.data import EppiePretrainDataset
from eppielm.models import (
    EppieLMConfig,
    EppieLMForCausalLM,
)
from eppielm.training import get_cosine_lr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def create_autocast_context(
    device: torch.device,
    precision: str,
):
    if device.type != "cuda":
        return nullcontext()

    if precision == "bf16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    if precision == "fp16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


def get_sdpa_context(
    backend: str,
):
    if backend == "auto":
        return nullcontext()

    backend_map = {
        "cudnn": (
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION
        ),
        "efficient": (
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
        ),
        "math": (
            torch.nn.attention.SDPBackend.MATH
        ),
        "flash": (
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ),
    }

    return torch.nn.attention.sdpa_kernel(
        backend_map[backend]
    )


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_optimizer(
    model: torch.nn.Module,
    args,
    device: torch.device,
):
    decay_params = []
    no_decay_params = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        if parameter.dim() >= 2:
            decay_params.append(parameter)
        else:
            no_decay_params.append(parameter)

    param_groups = [
        {
            "params": decay_params,
            "weight_decay": args.weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "lr": args.learning_rate,
        "betas": (
            args.beta1,
            args.beta2,
        ),
        "eps": args.eps,
    }

    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(
                param_groups,
                fused=True,
                **optimizer_kwargs,
            )

            return optimizer, True

        except (
            TypeError,
            RuntimeError,
        ):
            pass

    optimizer = torch.optim.AdamW(
        param_groups,
        **optimizer_kwargs,
    )

    return optimizer, False


def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(state) -> None:
    if not state:
        return

    if "python" in state:
        random.setstate(
            state["python"]
        )

    if "numpy" in state:
        np.random.set_state(
            state["numpy"]
        )

    if "torch" in state:
        torch.set_rng_state(
            state["torch"]
        )

    if (
        "cuda" in state
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            state["cuda"]
        )


def cleanup_old_checkpoints(
    output_dir: Path,
    keep_last: int,
) -> None:
    if keep_last <= 0:
        return

    checkpoints = []

    for path in output_dir.glob(
        "step_*"
    ):
        if not path.is_dir():
            continue

        try:
            step = int(
                path.name.split("_")[-1]
            )
        except ValueError:
            continue

        checkpoints.append(
            (step, path)
        )

    checkpoints.sort(
        key=lambda item: item[0]
    )

    excess = (
        len(checkpoints)
        - keep_last
    )

    if excess <= 0:
        return

    for _, path in checkpoints[:excess]:
        shutil.rmtree(
            path,
            ignore_errors=True,
        )

        print(
            f"[checkpoint] removed old: "
            f"{path}"
        )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    args,
    global_step: int,
    epoch: int,
    batch_idx: int,
    max_steps: int,
) -> Path:
    output_dir = Path(
        args.output_dir
    )

    checkpoint_dir = (
        output_dir
        / f"step_{global_step:08d}"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_state = {
        key: value.detach().cpu()
        for key, value
        in model.state_dict().items()
    }

    torch.save(
        model_state,
        checkpoint_dir
        / "model.pt",
    )

    trainer_state = {
        "optimizer": (
            optimizer.state_dict()
        ),
        "scaler": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "global_step": global_step,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "max_steps": max_steps,
        "seed": args.seed,
        "rng_state": get_rng_state(),
    }

    torch.save(
        trainer_state,
        checkpoint_dir
        / "trainer_state.pt",
    )

    with (
        checkpoint_dir
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            model.config.to_dict(),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[checkpoint] saved: "
        f"{checkpoint_dir}"
    )

    cleanup_old_checkpoints(
        output_dir=output_dir,
        keep_last=(
            args.keep_last_checkpoints
        ),
    )

    return checkpoint_dir


def find_latest_checkpoint(
    output_dir: str,
):
    output_dir = Path(
        output_dir
    )

    if not output_dir.exists():
        return None

    checkpoints = []

    for path in output_dir.glob(
        "step_*"
    ):
        if not path.is_dir():
            continue

        try:
            step = int(
                path.name.split("_")[-1]
            )
        except ValueError:
            continue

        model_path = (
            path
            / "model.pt"
        )

        trainer_path = (
            path
            / "trainer_state.pt"
        )

        if (
            model_path.exists()
            and trainer_path.exists()
        ):
            checkpoints.append(
                (step, path)
            )

    if not checkpoints:
        return None

    checkpoints.sort(
        key=lambda item: item[0]
    )

    return checkpoints[-1][1]


def load_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
):
    print(
        f"[checkpoint] loading: "
        f"{checkpoint_dir}"
    )

    model_state = torch.load(
        checkpoint_dir
        / "model.pt",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        model_state
    )

    trainer_state = torch.load(
        checkpoint_dir
        / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )

    optimizer.load_state_dict(
        trainer_state["optimizer"]
    )

    move_optimizer_state_to_device(
        optimizer,
        device,
    )

    scaler_state = trainer_state.get(
        "scaler"
    )

    if (
        scaler is not None
        and scaler_state
    ):
        scaler.load_state_dict(
            scaler_state
        )

    restore_rng_state(
        trainer_state.get(
            "rng_state"
        )
    )

    global_step = int(
        trainer_state[
            "global_step"
        ]
    )

    epoch = int(
        trainer_state[
            "epoch"
        ]
    )

    batch_idx = int(
        trainer_state[
            "batch_idx"
        ]
    )

    saved_max_steps = (
        trainer_state.get(
            "max_steps"
        )
    )

    if saved_max_steps is not None:
        saved_max_steps = int(
            saved_max_steps
        )

    return (
        global_step,
        epoch,
        batch_idx,
        saved_max_steps,
    )


def build_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    epoch: int,
):
    generator = torch.Generator()

    generator.manual_seed(
        seed + epoch
    )

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "generator": generator,
        "persistent_workers": (
            num_workers > 0
        ),
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = 4

    return DataLoader(
        **kwargs
    )


def train(args) -> None:
    set_seed(
        args.seed
    )

    if torch.cuda.is_available():
        device = torch.device(
            args.device
        )
    else:
        device = torch.device(
            "cpu"
        )

    if (
        device.type == "cuda"
        and args.precision == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "当前 GPU 不支持 BF16。"
        )

    if device.type == "cuda":
        torch.set_float32_matmul_precision(
            "high"
        )

    print("=" * 72)
    print("EppieLM-150M Pretraining")
    print("=" * 72)

    print(
        f"Device:                 "
        f"{device}"
    )

    if device.type == "cuda":
        props = (
            torch.cuda.get_device_properties(
                device
            )
        )

        print(
            f"GPU:                    "
            f"{props.name}"
        )

        print(
            f"VRAM:                   "
            f"{props.total_memory / 1024**3:.2f} GiB"
        )

    print(
        f"Precision:              "
        f"{args.precision}"
    )

    print(
        f"SDPA backend:           "
        f"{args.sdpa_backend}"
    )

    print(
        f"Data:                   "
        f"{args.data_path}"
    )

    print(
        f"Sequence length:        "
        f"{args.seq_len}"
    )

    print(
        f"Micro batch size:       "
        f"{args.micro_batch_size}"
    )

    print(
        f"Gradient accumulation:  "
        f"{args.gradient_accumulation_steps}"
    )

    dataset = EppiePretrainDataset(
        args.data_path,
        seq_len=args.seq_len,
        expected_vocab_size=15000,
    )

    print(
        f"Dataset:                "
        f"{dataset}"
    )

    effective_batch_sequences = (
        args.micro_batch_size
        * args.gradient_accumulation_steps
    )

    tokens_per_step = (
        effective_batch_sequences
        * args.seq_len
    )

    if args.max_steps <= 0:
        micro_batches_per_epoch = (
            len(dataset)
            // args.micro_batch_size
        )

        max_steps = (
            micro_batches_per_epoch
            // args.gradient_accumulation_steps
        )

    else:
        max_steps = args.max_steps

    if max_steps <= 0:
        raise RuntimeError(
            "max_steps 必须大于 0。"
        )

    effective_warmup_steps = min(
        args.warmup_steps,
        max_steps,
    )

    target_tokens = (
        max_steps
        * tokens_per_step
    )

    print(
        f"Effective batch:        "
        f"{effective_batch_sequences} sequences"
    )

    print(
        f"Tokens / step:          "
        f"{tokens_per_step:,}"
    )

    print(
        f"Max optimizer steps:    "
        f"{max_steps:,}"
    )

    print(
        f"Warmup steps:           "
        f"{effective_warmup_steps:,}"
    )

    print(
        f"Scheduled train tokens: "
        f"{target_tokens:,}"
    )

    config = EppieLMConfig(
        vocab_size=15000,
        hidden_size=args.hidden_size,
        intermediate_size=(
            args.intermediate_size
        ),
        num_hidden_layers=(
            args.num_hidden_layers
        ),
        num_attention_heads=(
            args.num_attention_heads
        ),
        num_key_value_heads=(
            args.num_key_value_heads
        ),
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        initializer_range=0.02,
        attention_impl=(
            args.attention_impl
        ),
        use_cache=False,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )

    model = EppieLMForCausalLM(
        config
    )

    num_parameters = count_parameters(
        model
    )

    print(
        f"Parameters:             "
        f"{num_parameters / 1e6:.2f}M"
    )

    model = model.to(
        device
    )

    # raw_model 始终用于 optimizer、checkpoint、grad clipping。
    # compiled model 只负责 forward/backward。
    raw_model = model

    if args.gradient_checkpointing:
        raw_model.gradient_checkpointing_enable()

        print(
            "Gradient checkpointing: enabled"
        )
    else:
        print(
            "Gradient checkpointing: disabled"
        )

    optimizer, fused_optimizer = (
        build_optimizer(
            model=raw_model,
            args=args,
            device=device,
        )
    )

    print(
        f"Fused AdamW:            "
        f"{fused_optimizer}"
    )

    use_fp16_scaler = (
        device.type == "cuda"
        and args.precision == "fp16"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_fp16_scaler,
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_step = 0
    start_epoch = 0
    resume_batch_idx = 0
    last_saved_step = -1

    if args.resume:
        checkpoint_dir = (
            find_latest_checkpoint(
                args.output_dir
            )
        )

        if checkpoint_dir is None:
            print(
                "[checkpoint] no checkpoint found, "
                "starting from scratch"
            )

        else:
            (
                global_step,
                start_epoch,
                resume_batch_idx,
                saved_max_steps,
            ) = load_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
            )

            if (
                saved_max_steps is not None
                and saved_max_steps
                != max_steps
            ):
                raise RuntimeError(
                    "resume 时 max_steps 与 "
                    "checkpoint 不一致："
                    f"{saved_max_steps} != {max_steps}"
                )

            last_saved_step = (
                global_step
            )

            print(
                f"[checkpoint] resumed "
                f"step={global_step:,}, "
                f"epoch={start_epoch}, "
                f"batch={resume_batch_idx:,}"
            )

    if global_step >= max_steps:
        print(
            "Checkpoint 已达到训练目标，"
            "无需继续训练。"
        )

        return

    print(
        f"torch.compile:          "
        f"{args.compile}"
    )

    if args.compile:
        if device.type != "cuda":
            raise RuntimeError(
                "当前训练配置仅在 CUDA 上启用 torch.compile。"
            )

        print(
            f"Compile mode:           "
            f"{args.compile_mode}"
        )

        print(
            "Compiling model "
            "(first forward may take a while)..."
        )

        model = torch.compile(
            raw_model,
            mode=args.compile_mode,
            fullgraph=False,
            dynamic=False,
        )

    else:
        model = raw_model

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    epoch = start_epoch

    session_start_time = (
        time.perf_counter()
    )

    log_start_time = (
        time.perf_counter()
    )

    # 累计 token 用于显示总体训练进度。
    cumulative_processed_tokens = (
        global_step
        * tokens_per_step
    )

    # session token 只统计本次进程真正训练的 token，
    # 用于计算 resume 后正确的平均吞吐。
    session_processed_tokens = 0

    log_tokens = 0
    log_loss = 0.0
    log_updates = 0

    running_loss = 0.0
    running_micro_steps = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            device
        )

    with get_sdpa_context(
        args.sdpa_backend
    ):
        while global_step < max_steps:
            loader = build_dataloader(
                dataset=dataset,
                batch_size=(
                    args.micro_batch_size
                ),
                num_workers=(
                    args.num_workers
                ),
                pin_memory=(
                    device.type == "cuda"
                ),
                seed=args.seed,
                epoch=epoch,
            )

            if len(loader) == 0:
                raise RuntimeError(
                    "DataLoader 为空。"
                )

            for batch_idx, (
                input_ids,
                labels,
            ) in enumerate(loader):
                if (
                    epoch == start_epoch
                    and batch_idx
                    < resume_batch_idx
                ):
                    continue

                input_ids = input_ids.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                with create_autocast_context(
                    device,
                    args.precision,
                ):
                    outputs = model(
                        input_ids=input_ids,
                        labels=labels,
                        use_cache=False,
                        return_dict=True,
                    )

                    loss = outputs.loss

                    if not torch.isfinite(
                        loss
                    ):
                        raise RuntimeError(
                            "检测到非有限 loss: "
                            f"{loss.item()}"
                        )

                    scaled_loss = (
                        loss
                        / args.gradient_accumulation_steps
                    )

                if use_fp16_scaler:
                    scaler.scale(
                        scaled_loss
                    ).backward()

                else:
                    scaled_loss.backward()

                running_loss += (
                    loss.detach()
                    .float()
                    .item()
                )

                running_micro_steps += 1

                should_update = (
                    running_micro_steps
                    >= args.gradient_accumulation_steps
                )

                if not should_update:
                    continue

                lr = get_cosine_lr(
                    step=global_step,
                    max_steps=max_steps,
                    warmup_steps=(
                        effective_warmup_steps
                    ),
                    max_lr=(
                        args.learning_rate
                    ),
                    min_lr=(
                        args.min_learning_rate
                    ),
                )

                set_optimizer_lr(
                    optimizer,
                    lr,
                )

                if use_fp16_scaler:
                    scaler.unscale_(
                        optimizer
                    )

                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        raw_model.parameters(),
                        args.max_grad_norm,
                    )
                )

                if not torch.isfinite(
                    grad_norm
                ):
                    raise RuntimeError(
                        "检测到非有限 gradient norm: "
                        f"{float(grad_norm)}"
                    )

                if use_fp16_scaler:
                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:
                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

                step_loss = (
                    running_loss
                    / running_micro_steps
                )

                running_loss = 0.0
                running_micro_steps = 0

                cumulative_processed_tokens += (
                    tokens_per_step
                )

                session_processed_tokens += (
                    tokens_per_step
                )

                log_tokens += (
                    tokens_per_step
                )

                log_loss += (
                    step_loss
                )

                log_updates += 1

                should_log = (
                    global_step == 1
                    or global_step
                    % args.log_steps
                    == 0
                )

                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize(
                            device
                        )

                    now = (
                        time.perf_counter()
                    )

                    log_elapsed = (
                        now
                        - log_start_time
                    )

                    throughput = (
                        log_tokens
                        / log_elapsed
                        if log_elapsed > 0
                        else 0.0
                    )

                    mean_log_loss = (
                        log_loss
                        / log_updates
                    )

                    progress = (
                        global_step
                        / max_steps
                    )

                    session_elapsed = (
                        now
                        - session_start_time
                    )

                    completed_this_session = (
                        global_step
                        - (
                            cumulative_processed_tokens
                            - session_processed_tokens
                        )
                        // tokens_per_step
                    )

                    remaining_steps = (
                        max_steps
                        - global_step
                    )

                    session_steps = (
                        session_processed_tokens
                        // tokens_per_step
                    )

                    if session_steps > 0:
                        seconds_per_step = (
                            session_elapsed
                            / session_steps
                        )

                        eta_seconds = (
                            remaining_steps
                            * seconds_per_step
                        )
                    else:
                        eta_seconds = 0.0

                    if device.type == "cuda":
                        peak_vram = (
                            torch.cuda.max_memory_allocated(
                                device
                            )
                            / 1024**3
                        )
                    else:
                        peak_vram = 0.0

                    print(
                        f"step "
                        f"{global_step:7,d}/"
                        f"{max_steps:,} | "
                        f"{progress * 100:6.2f}% | "
                        f"loss {mean_log_loss:.4f} | "
                        f"lr {lr:.3e} | "
                        f"grad {float(grad_norm):.3f} | "
                        f"tok/s {throughput:,.0f} | "
                        f"peak {peak_vram:.2f} GiB | "
                        f"ETA {eta_seconds / 3600:.1f}h"
                    )

                    log_start_time = (
                        time.perf_counter()
                    )

                    log_tokens = 0
                    log_loss = 0.0
                    log_updates = 0

                if (
                    args.save_steps > 0
                    and global_step
                    % args.save_steps
                    == 0
                ):
                    save_checkpoint(
                        model=raw_model,
                        optimizer=optimizer,
                        scaler=scaler,
                        args=args,
                        global_step=global_step,
                        epoch=epoch,
                        batch_idx=(
                            batch_idx + 1
                        ),
                        max_steps=max_steps,
                    )

                    last_saved_step = (
                        global_step
                    )

                    # checkpoint I/O 不计入下一段局部 tok/s。
                    if device.type == "cuda":
                        torch.cuda.synchronize(
                            device
                        )

                    log_start_time = (
                        time.perf_counter()
                    )

                    log_tokens = 0
                    log_loss = 0.0
                    log_updates = 0

                if (
                    global_step
                    >= max_steps
                ):
                    break

            if running_micro_steps > 0:
                optimizer.zero_grad(
                    set_to_none=True
                )

                running_loss = 0.0
                running_micro_steps = 0

            epoch += 1
            resume_batch_idx = 0

    if last_saved_step != global_step:
        final_checkpoint = (
            save_checkpoint(
                model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                global_step=global_step,
                epoch=epoch,
                batch_idx=0,
                max_steps=max_steps,
            )
        )

    else:
        final_checkpoint = (
            Path(args.output_dir)
            / f"step_{global_step:08d}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    session_total_time = (
        time.perf_counter()
        - session_start_time
    )

    average_throughput = (
        session_processed_tokens
        / session_total_time
        if session_total_time > 0
        else 0.0
    )

    print()
    print("=" * 72)
    print("Training finished")
    print("=" * 72)

    print(
        f"Final step:             "
        f"{global_step:,}"
    )

    print(
        f"Cumulative tokens:      "
        f"{cumulative_processed_tokens:,}"
    )

    print(
        f"Session tokens:         "
        f"{session_processed_tokens:,}"
    )

    print(
        f"Session training time:  "
        f"{session_total_time / 3600:.2f} h"
    )

    print(
        f"Session average tok/s:  "
        f"{average_throughput:,.0f}"
    )

    print(
        f"Final checkpoint:       "
        f"{final_checkpoint}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "EppieLM single-GPU pretraining"
        )
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            r"E:\EppieLMData\outputs"
            r"\eppielm_150m_3b"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--precision",
        type=str,
        choices=[
            "bf16",
            "fp16",
            "fp32",
        ],
        default="bf16",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
    )

    # 0 = 自动完整走当前数据集一遍。
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )

    # Windows + memmap 下默认 0 更稳。
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--log_steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--keep_last_checkpoints",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )

    parser.add_argument(
        "--sdpa_backend",
        type=str,
        choices=[
            "auto",
            "cudnn",
            "efficient",
            "math",
            "flash",
        ],
        default="cudnn",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile.",
    )

    parser.add_argument(
        "--compile_mode",
        type=str,
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
        ],
        default="default",
        help="torch.compile mode.",
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
    )

    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=22,
    )

    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--num_key_value_heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--attention_impl",
        type=str,
        choices=[
            "sdpa",
            "eager",
        ],
        default="sdpa",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)