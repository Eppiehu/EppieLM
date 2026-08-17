from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .attention import EppieAttention, PastKeyValue
from .configuration_eppielm import EppieLMConfig
from .mlp import EppieMLP
from .norm import EppieRMSNorm
from .rope import EppieRotaryEmbedding


PastKeyValues = List[Optional[PastKeyValue]]


class EppieLMPreTrainedModel(PreTrainedModel):
    config_class = EppieLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["EppieDecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.initializer_range,
            )

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.initializer_range,
            )

            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class EppieDecoderLayer(nn.Module):
    def __init__(
        self,
        config: EppieLMConfig,
        layer_idx: int,
    ):
        super().__init__()

        self.layer_idx = layer_idx

        self.input_layernorm = EppieRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.self_attn = EppieAttention(config)

        self.post_attention_layernorm = EppieRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.mlp = EppieMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[PastKeyValue],
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(
            hidden_states
        )

        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(
            hidden_states
        )

        hidden_states = self.mlp(
            hidden_states
        )

        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class EppieLMModel(EppieLMPreTrainedModel):
    def __init__(self, config: EppieLMConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        self.dropout = nn.Dropout(
            config.hidden_dropout
        )

        self.layers = nn.ModuleList(
            [
                EppieDecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(
                    config.num_hidden_layers
                )
            ]
        )

        self.norm = EppieRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.rotary_emb = EppieRotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=(
                config.max_position_embeddings
            ),
            base=config.rope_theta,
        )

        self.gradient_checkpointing = False

        # 1 = checkpoint 每一层。
        # N > 1 = 每 N 层 checkpoint 一层，减少重算、增加显存。
        self.checkpoint_every = 1

        # 当前 EppieLM baseline dropout=0，因此 checkpoint
        # 重算时不需要保存/恢复 RNG 状态。
        self.checkpoint_preserve_rng_state = bool(
            config.hidden_dropout > 0.0
            or config.attention_dropout > 0.0
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def set_checkpoint_every(
        self,
        value: int,
    ) -> None:
        value = int(value)

        if value < 1:
            raise ValueError(
                "checkpoint_every must be >= 1"
            )

        self.checkpoint_every = value

    def get_checkpointed_layer_count(
        self,
    ) -> int:
        if not self.gradient_checkpointing:
            return 0

        return sum(
            1
            for layer_idx in range(
                len(self.layers)
            )
            if layer_idx % self.checkpoint_every == 0
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PastKeyValues] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        Tuple,
        BaseModelOutputWithPast,
    ]:
        use_cache = (
            self.config.use_cache
            if use_cache is None
            else use_cache
        )

        return_dict = (
            self.config.return_dict
            if return_dict is None
            else return_dict
        )

        batch_size, seq_len = input_ids.shape

        if past_key_values is None:
            past_key_values = [
                None
            ] * len(self.layers)

        if len(past_key_values) != len(self.layers):
            raise ValueError(
                "past_key_values must contain one entry "
                "per decoder layer."
            )

        past_len = 0

        first_cache = past_key_values[0]

        if first_cache is not None:
            past_len = first_cache[0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_len,
                past_len + seq_len,
                device=input_ids.device,
                dtype=torch.long,
            )

            position_ids = position_ids.unsqueeze(0).expand(
                batch_size,
                -1,
            )

        if position_ids.max().item() >= (
            self.config.max_position_embeddings
        ):
            raise ValueError(
                "Sequence position exceeds "
                "max_position_embeddings="
                f"{self.config.max_position_embeddings}."
            )

        hidden_states = self.embed_tokens(
            input_ids
        )

        hidden_states = self.dropout(
            hidden_states
        )

        cos, sin = self.rotary_emb(
            position_ids=position_ids,
            dtype=hidden_states.dtype,
        )

        next_cache = [] if use_cache else None

        for layer_idx, decoder_layer in enumerate(
            self.layers
        ):
            layer_past = past_key_values[layer_idx]

            should_checkpoint = (
                self.gradient_checkpointing
                and self.training
                and (
                    layer_idx
                    % self.checkpoint_every
                    == 0
                )
            )

            if should_checkpoint:
                if use_cache:
                    raise ValueError(
                        "use_cache=True is incompatible "
                        "with gradient checkpointing "
                        "during training."
                    )

                # 显式绑定当前 decoder_layer。
                # checkpoint backward 会稍后重新调用该函数，
                # 不能依赖循环变量的晚绑定行为。
                def custom_forward(
                    states,
                    layer=decoder_layer,
                ):
                    return layer(
                        hidden_states=states,
                        cos=cos,
                        sin=sin,
                        attention_mask=attention_mask,
                        past_key_value=None,
                        use_cache=False,
                    )[0]

                hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        custom_forward,
                        hidden_states,
                        use_reentrant=False,
                        preserve_rng_state=(
                            self.checkpoint_preserve_rng_state
                        ),
                    )
                )

                present_key_value = None

            else:
                (
                    hidden_states,
                    present_key_value,
                ) = decoder_layer(
                    hidden_states=hidden_states,
                    cos=cos,
                    sin=sin,
                    attention_mask=attention_mask,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )

            if use_cache:
                next_cache.append(
                    present_key_value
                )

        hidden_states = self.norm(
            hidden_states
        )

        if not return_dict:
            return (
                hidden_states,
                next_cache,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
        )


class EppieLMForCausalLM(EppieLMPreTrainedModel):
    _tied_weights_keys = {
        "lm_head.weight": "model.embed_tokens.weight"
    }

    def __init__(self, config: EppieLMConfig):
        super().__init__(config)

        self.model = EppieLMModel(config)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.post_init()

        if config.tie_word_embeddings:
            self.tie_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(
        self,
        new_embeddings,
    ):
        self.lm_head = new_embeddings

    def set_checkpoint_every(
        self,
        value: int,
    ) -> None:
        self.model.set_checkpoint_every(
            value
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PastKeyValues] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int = 0,
        return_dict: Optional[bool] = None,
    ) -> Union[
        Tuple,
        CausalLMOutputWithPast,
    ]:
        return_dict = (
            self.config.return_dict
            if return_dict is None
            else return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        if logits_to_keep > 0:
            logits = self.lm_head(
                hidden_states[
                    :,
                    -logits_to_keep:,
                    :,
                ]
            )
        else:
            logits = self.lm_head(
                hidden_states
            )

        # Transformer body 可以保持 BF16，
        # logits/loss 转 FP32 保证训练稳定性。
        logits = logits.float()

        loss = None

        if labels is not None:
            if logits_to_keep > 0:
                raise ValueError(
                    "logits_to_keep must be 0 "
                    "when labels are provided."
                )

            shift_logits = (
                logits[:, :-1, :]
                .contiguous()
            )

            shift_labels = (
                labels[:, 1:]
                .contiguous()
            )

            loss = F.cross_entropy(
                shift_logits.view(
                    -1,
                    self.config.vocab_size,
                ),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            return (
                loss,
                logits,
                outputs.past_key_values,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=(
                outputs.past_key_values
            ),
            hidden_states=hidden_states,
        )