import math


def get_cosine_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Linear warmup + cosine decay。

    step 表示 optimizer step，而不是 micro batch step。
    """
    if step < 0:
        raise ValueError("step 必须 >= 0")

    if max_steps <= 0:
        raise ValueError("max_steps 必须 > 0")

    if warmup_steps < 0:
        raise ValueError("warmup_steps 必须 >= 0")

    if warmup_steps > max_steps:
        raise ValueError(
            "warmup_steps 不能大于 max_steps"
        )

    if max_lr <= 0:
        raise ValueError("max_lr 必须 > 0")

    if min_lr < 0:
        raise ValueError("min_lr 必须 >= 0")

    if min_lr > max_lr:
        raise ValueError(
            "min_lr 不能大于 max_lr"
        )

    # 第一个 optimizer step 不从严格的 0 LR 开始，
    # 避免第一步完全不更新参数。
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (
            (step + 1) / warmup_steps
        )

    if step >= max_steps:
        return min_lr

    if max_steps == warmup_steps:
        return max_lr

    progress = (
        step - warmup_steps
    ) / (
        max_steps - warmup_steps
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi * progress
        )
    )

    return (
        min_lr
        + cosine
        * (max_lr - min_lr)
    )