import torch
import torch.nn.functional as F


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = "cuda"
    dtype = torch.bfloat16

    print("=" * 72)
    print("SDPA BACKEND CHECK")
    print("=" * 72)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

    batch_size = 1
    num_heads = 12
    seq_len = 2048
    head_dim = 64

    q = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
    )

    k = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
    )

    v = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
    )

    print()
    print("Test shape:")
    print(f"Q: {tuple(q.shape)}")
    print(f"K: {tuple(k.shape)}")
    print(f"V: {tuple(v.shape)}")

    print()
    print("CUDA SDPA settings:")
    print(
        "Flash SDP enabled:",
        torch.backends.cuda.flash_sdp_enabled(),
    )
    print(
        "Memory-efficient SDP enabled:",
        torch.backends.cuda.mem_efficient_sdp_enabled(),
    )
    print(
        "Math SDP enabled:",
        torch.backends.cuda.math_sdp_enabled(),
    )
    print(
        "cuDNN SDP enabled:",
        torch.backends.cuda.cudnn_sdp_enabled(),
    )

    print()
    print("Running normal SDPA...")

    torch.cuda.synchronize()

    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=True,
    )

    torch.cuda.synchronize()

    print("Normal SDPA: OK")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype}")

    print()
    print("=" * 72)
    print("FORCED BACKEND TEST")
    print("=" * 72)

    backends = [
        ("FLASH_ATTENTION", torch.nn.attention.SDPBackend.FLASH_ATTENTION),
        (
            "EFFICIENT_ATTENTION",
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        ),
        ("MATH", torch.nn.attention.SDPBackend.MATH),
        ("CUDNN_ATTENTION", torch.nn.attention.SDPBackend.CUDNN_ATTENTION),
    ]

    for name, backend in backends:
        print()
        print(f"Testing {name}...")

        try:
            with torch.nn.attention.sdpa_kernel(
                backend
            ):
                torch.cuda.synchronize()

                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    is_causal=True,
                )

                torch.cuda.synchronize()

            print(f"{name}: AVAILABLE")
            print(f"shape={tuple(out.shape)}")

        except Exception as e:
            print(f"{name}: NOT AVAILABLE")
            print(
                f"{type(e).__name__}: {e}"
            )


if __name__ == "__main__":
    main()