import math

import torch
from torch.optim import AdamW

from eppielm.models import (
    EppieLMConfig,
    EppieLMForCausalLM,
)
from eppielm.training import get_cosine_lr


def tiny_config():
    return EppieLMConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_impl="sdpa",
        use_cache=False,
    )


def test_lr_warmup():
    lr = get_cosine_lr(
        step=0,
        max_steps=100,
        warmup_steps=10,
        max_lr=1e-3,
        min_lr=1e-4,
    )

    assert math.isclose(
        lr,
        1e-4,
        rel_tol=1e-6,
    )


def test_lr_reaches_max_after_warmup():
    lr = get_cosine_lr(
        step=10,
        max_steps=100,
        warmup_steps=10,
        max_lr=1e-3,
        min_lr=1e-4,
    )

    assert math.isclose(
        lr,
        1e-3,
        rel_tol=1e-6,
    )


def test_lr_decays():
    lr_early = get_cosine_lr(
        step=20,
        max_steps=100,
        warmup_steps=10,
        max_lr=1e-3,
        min_lr=1e-4,
    )

    lr_late = get_cosine_lr(
        step=80,
        max_steps=100,
        warmup_steps=10,
        max_lr=1e-3,
        min_lr=1e-4,
    )

    assert (
        lr_late
        < lr_early
    )


def test_lr_reaches_min():
    lr = get_cosine_lr(
        step=100,
        max_steps=100,
        warmup_steps=10,
        max_lr=1e-3,
        min_lr=1e-4,
    )

    assert math.isclose(
        lr,
        1e-4,
        rel_tol=1e-6,
    )


def test_training_step_updates_weights():
    torch.manual_seed(
        42
    )

    model = EppieLMForCausalLM(
        tiny_config()
    )

    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    input_ids = torch.randint(
        low=0,
        high=256,
        size=(
            2,
            32,
        ),
    )

    labels = input_ids.clone()

    before = (
        model.model.embed_tokens.weight
        .detach()
        .clone()
    )

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )

    loss = outputs.loss

    assert torch.isfinite(
        loss
    )

    loss.backward()

    grad_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
    )

    assert torch.isfinite(
        grad_norm
    )

    optimizer.step()

    optimizer.zero_grad(
        set_to_none=True
    )

    after = (
        model.model.embed_tokens.weight
        .detach()
    )

    assert not torch.equal(
        before,
        after,
    )


def test_two_training_steps_reduce_or_keep_finite_loss():
    torch.manual_seed(
        42
    )

    model = EppieLMForCausalLM(
        tiny_config()
    )

    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    input_ids = torch.randint(
        0,
        256,
        (
            2,
            32,
        ),
    )

    labels = input_ids.clone()

    losses = []

    for _ in range(2):
        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

        loss = outputs.loss

        assert torch.isfinite(
            loss
        )

        losses.append(
            loss.item()
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

    # 不要求随机初始化模型每一步严格单调，
    # 这里只检查优化后仍然处于正常数值范围。
    assert all(
        math.isfinite(loss)
        for loss in losses
    )