"""Model configuration for EppieLM."""

from transformers import PretrainedConfig


class EppieLMConfig(PretrainedConfig):
    """
    EppieLM 模型配置。

    默认参数对应 EppieLM-150M。
    """

    model_type = "eppielm"

    def __init__(
        self,
        vocab_size: int = 15000,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        initializer_range: float = 0.02,
        attention_impl: str = "sdpa",
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.max_position_embeddings = max_position_embeddings

        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        self.initializer_range = initializer_range

        self.attention_impl = attention_impl

        self.use_cache = use_cache

        self._validate()

    @property
    def head_dim(self) -> int:
        return (
            self.hidden_size
            // self.num_attention_heads
        )

    @property
    def num_key_value_groups(self) -> int:
        return (
            self.num_attention_heads
            // self.num_key_value_heads
        )

    def _validate(self) -> None:
        # 注意力头必须能够精确划分 hidden states，
        # 否则后续 reshape 会隐藏真正的配置错误。
        if (
            self.hidden_size
            % self.num_attention_heads
            != 0
        ):
            raise ValueError(
                "hidden_size must be divisible by "
                "num_attention_heads, "
                f"got hidden_size={self.hidden_size} "
                f"and num_attention_heads="
                f"{self.num_attention_heads}."
            )

        # GQA 中每个 KV head 应服务相同数量的 Query heads。
        if (
            self.num_attention_heads
            % self.num_key_value_heads
            != 0
        ):
            raise ValueError(
                "num_attention_heads must be divisible "
                "by num_key_value_heads for grouped-query "
                "attention, "
                f"got {self.num_attention_heads} and "
                f"{self.num_key_value_heads}."
            )

        if (
            self.num_key_value_heads
            > self.num_attention_heads
        ):
            raise ValueError(
                "num_key_value_heads cannot exceed "
                "num_attention_heads."
            )

        if self.vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive."
            )

        if self.hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive."
            )

        if self.intermediate_size <= 0:
            raise ValueError(
                "intermediate_size must be positive."
            )

        if self.num_hidden_layers <= 0:
            raise ValueError(
                "num_hidden_layers must be positive."
            )

        if self.max_position_embeddings <= 0:
            raise ValueError(
                "max_position_embeddings must be positive."
            )

        if self.rms_norm_eps <= 0:
            raise ValueError(
                "rms_norm_eps must be positive."
            )

        if self.rope_theta <= 0:
            raise ValueError(
                "rope_theta must be positive."
            )

        if self.attention_dropout < 0:
            raise ValueError(
                "attention_dropout cannot be negative."
            )

        if self.hidden_dropout < 0:
            raise ValueError(
                "hidden_dropout cannot be negative."
            )

        if self.attention_impl not in {
            "sdpa",
            "eager",
        }:
            raise ValueError(
                "attention_impl must be either "
                "'sdpa' or 'eager'."
            )