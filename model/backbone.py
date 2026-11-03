"""FatigueGuard attention, modality experts, shared experts, and fusion encoder."""

import math
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from .config import FatigueGuardConfig

logger = logging.get_logger(__name__)


class SeqFusion(nn.Module):
    """Fuse a modality expert with the shared expert.

    ``use_gate=False`` is the controlled gate ablation: both experts remain
    active, but their outputs receive fixed equal weights instead of learned
    token-wise weights.
    """

    def __init__(self, dim, use_gate=True):
        super().__init__()
        self.use_gate = use_gate
        if use_gate:
            self.weight = nn.Parameter(torch.randn(dim, 1))
        else:
            self.register_parameter("weight", None)
        self.softmax = nn.Softmax(1)

    def forward(self, seq1, seq2):
        if not self.use_gate:
            return 0.5 * (seq1 + seq2)
        o1 = seq1 @ self.weight
        o2 = seq2 @ self.weight
        o = torch.cat([o1, o2], dim=-1)
        alpha = self.softmax(o)
        seq1 = seq1 * alpha[:, :, 0].unsqueeze(-1)
        seq2 = seq2 * alpha[:, :, 1].unsqueeze(-1)
        out = seq1 + seq2
        return out


class FatigueGuardPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FatigueGuardConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if (
            isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Embedding)
            or isinstance(module, nn.Linear)
        ):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class MultiHeadAttention(nn.Module):

    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and (
            not hasattr(config, "embedding_size")
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        query_length,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = (key_layer, value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)
        attention_probs_dropped = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask
        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        outputs = outputs + (past_key_value,)
        return outputs


class SelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):

    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.attention = MultiHeadAttention(config, is_cross_attention)
        self.output = SelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.attention.num_attention_heads,
            self.attention.attention_head_size,
            self.pruned_heads,
        )
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(
            heads
        )
        self.attention.all_head_size = (
            self.attention.attention_head_size * self.attention.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_length,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.attention(
            hidden_states,
            query_length,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class Intermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Layer(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Attention(config)
        self.layer_idx = layer_idx
        self.has_cross_attention = False
        if self.layer_idx == self.config.num_hidden_layers - 1:
            self.intermediate = Intermediate(config)
            self.output = Output(config)
        else:
            self.intermediate_eog = Intermediate(config)
            self.output_eog = Output(config)
            self.intermediate_eeg = Intermediate(config)
            self.output_eeg = Output(config)
            self.intermediate_fusion = Intermediate(config)
            self.output_fusion = Output(config)
            self.seq_fusion = SeqFusion(
                config.hidden_size, use_gate=not getattr(config, "disable_gate", False)
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
        eog_first=True,
        has_eog=True,
        has_eeg=True,
        mode="mat",
    ):
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            query_length,
            attention_mask,
            head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]
        if self.layer_idx == self.config.num_hidden_layers - 1:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        elif has_eog and has_eeg:
            if eog_first:
                eog_attention_output = attention_output[:, :query_length, :]
                eeg_attention_output = attention_output[:, query_length:, :]
            else:
                eeg_attention_output = attention_output[:, :query_length, :]
                eog_attention_output = attention_output[:, query_length:, :]
            layer_output_eog = apply_chunking_to_forward(
                self.feed_forward_chunk_eog,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                eog_attention_output,
            )
            layer_output_eeg = apply_chunking_to_forward(
                self.feed_forward_chunk_eeg,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                eeg_attention_output,
            )
            use_shared_expert = not getattr(
                self.config, "disable_shared_expert", False
            ) and (mode == "mat" or mode == "gen" or mode == "ft")
            if use_shared_expert:
                layer_output_fusion = apply_chunking_to_forward(
                    self.feed_forward_chunk_fusion,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output,
                )
                if eog_first:
                    layer_output_tmp = torch.cat(
                        [layer_output_eog, layer_output_eeg], dim=1
                    )
                else:
                    layer_output_tmp = torch.cat(
                        [layer_output_eeg, layer_output_eog], dim=1
                    )
                layer_output = self.seq_fusion(layer_output_tmp, layer_output_fusion)
            elif eog_first:
                layer_output = torch.cat([layer_output_eog, layer_output_eeg], dim=1)
            else:
                layer_output = torch.cat([layer_output_eeg, layer_output_eog], dim=1)
        elif has_eeg and (not has_eog):
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_eeg,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
            use_shared_expert = not getattr(
                self.config, "disable_shared_expert", False
            ) and (mode == "mat" or mode == "gen" or mode == "ft")
            if use_shared_expert:
                layer_output_fusion = apply_chunking_to_forward(
                    self.feed_forward_chunk_fusion,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output,
                )
                layer_output = self.seq_fusion(layer_output, layer_output_fusion)
        elif has_eog and (not has_eeg):
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_eog,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
            use_shared_expert = not getattr(
                self.config, "disable_shared_expert", False
            ) and (mode == "mat" or mode == "gen" or mode == "ft")
            if use_shared_expert:
                layer_output_fusion = apply_chunking_to_forward(
                    self.feed_forward_chunk_fusion,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output,
                )
                layer_output = self.seq_fusion(layer_output, layer_output_fusion)
        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk_eeg(self, attention_output):
        intermediate_output = self.intermediate_eeg(attention_output)
        layer_output = self.output_eeg(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_eog(self, attention_output):
        intermediate_output = self.intermediate_eog(attention_output)
        layer_output = self.output_eog(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_fusion(self, attention_output):
        intermediate_output = self.intermediate_fusion(attention_output)
        layer_output = self.output_fusion(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class ExpertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [Layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
        eog_first=True,
        has_eog=True,
        has_eeg=True,
        mode="mat",
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        layer_count = self.config.num_hidden_layers
        if getattr(self.config, "disable_fusion_transformer", False):
            layer_count -= 1
        if layer_count < 1:
            raise ValueError("At least one expert layer is required")
        for i in range(layer_count):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    eog_first=eog_first,
                    has_eog=has_eog,
                    has_eeg=has_eeg,
                    mode=mode,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                    eog_first=eog_first,
                    has_eog=has_eog,
                    has_eeg=has_eeg,
                    mode=mode,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                (
                    v
                    for v in [
                        hidden_states,
                        next_decoder_cache,
                        all_hidden_states,
                        all_self_attentions,
                        all_cross_attentions,
                    ]
                    if v is not None
                )
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class FatigueGuardBackbone(FatigueGuardPreTrainedModel):
    """
    Multimodal feature encoder with task-specific attention masks.
    """

    def __init__(self, config: FatigueGuardConfig):
        super().__init__(config)
        self.config = config
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = ExpertEncoder(config)
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        pre_text_length: int,
        text_length: int,
        device: torch.device,
        mode="query",
        has_query: bool = False,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        seq_length = pre_text_length + text_length
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if mode == "mat" or mode == "ft":
                extended_attention_mask = attention_mask[:, None, None, :]
            elif mode == "con":
                query_mask = torch.ones(
                    (pre_text_length, pre_text_length), device=device
                )
                text_mask = torch.ones((text_length, text_length), device=device)
                full_mask = torch.zeros(
                    (batch_size, seq_length, seq_length), device=device
                )
                full_mask[:, :pre_text_length, :pre_text_length] = query_mask
                full_mask[:, pre_text_length:, pre_text_length:] = text_mask
                extended_attention_mask = (
                    full_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            elif mode == "gen":
                text_ids = torch.arange(pre_text_length, seq_length, device=device)
                causal_mask = (
                    text_ids[None, None, :].repeat(batch_size, text_length, 1)
                    <= text_ids[None, :, None]
                )
                causal_mask = causal_mask.to(attention_mask.dtype)
                causal_mask = torch.cat(
                    [
                        torch.zeros(
                            (batch_size, pre_text_length, text_length),
                            device=device,
                            dtype=causal_mask.dtype,
                        ),
                        causal_mask,
                    ],
                    dim=1,
                )
                causal_mask = torch.cat(
                    [
                        torch.ones(
                            (batch_size, causal_mask.shape[1], pre_text_length),
                            device=device,
                            dtype=causal_mask.dtype,
                        ),
                        causal_mask,
                    ],
                    dim=-1,
                )
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                raise ValueError("mode should be either 'con', 'mat' or 'gen'")
        else:
            raise ValueError(
                "Wrong shape for attention_mask (shape {})".format(attention_mask.shape)
            )
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        query_embeds: torch.FloatTensor,
        mode: Optional[str] = "query",
        text_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        eog_first=True,
        has_eog=True,
        has_eeg=True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        use_cache (`bool`, `optional`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        pre_text_length = query_length = (
            query_embeds.shape[1] if query_embeds is not None else 0
        )
        text_length = text_embeds.shape[1] if text_embeds is not None else 0
        if query_embeds is not None and text_embeds is not None:
            full_embeds = torch.cat([query_embeds, text_embeds], dim=1)
        elif query_embeds is not None:
            full_embeds = query_embeds
        elif text_embeds is not None:
            full_embeds = text_embeds
        else:
            raise ValueError("query_embeds or text_embeds should be provided")
        embedding_output = self.layernorm(full_embeds)
        embedding_output = self.dropout(embedding_output)
        batch_size, _ = embedding_output.size()[:-1]
        device = embedding_output.device
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, full_embeds.shape[1]), device=device
            )
        elif attention_mask.size()[1] == text_length:
            attention_mask = torch.cat(
                [
                    torch.ones((batch_size, pre_text_length), device=device),
                    attention_mask,
                ],
                dim=1,
            )
        elif attention_mask.size()[1] != text_length + pre_text_length:
            raise ValueError(
                f"attention_mask has the wrong size, got {attention_mask.size()[1]}, should be {text_length} or {text_length + pre_text_length}"
            )
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, batch_size, pre_text_length, text_length, device, mode
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
            eog_first=eog_first,
            has_eog=has_eog,
            has_eeg=has_eeg,
            mode=mode,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
