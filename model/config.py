"""Configuration for the FatigueGuard feature encoder."""

from transformers import PretrainedConfig


EXPERIMENTS = (
    "full",
    "scratch",
    "no_con",
    "no_mat",
    "no_gen",
    "no_shared_expert",
    "no_gate",
    "no_fusion_transformer",
)


class FatigueGuardConfig(PretrainedConfig):
    model_type = "fatigueguard"

    def __init__(self, **kwargs):
        defaults = dict(
            hidden_size=64,
            intermediate_size=256,
            num_attention_heads=4,
            num_hidden_layers=4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            chunk_size_feed_forward=0,
            position_embedding_type="absolute",
            max_position_embeddings=512,
            encoder_hidden_size=64,
            if_DDP=False,
            output_hidden_states=True,
            disable_shared_expert=False,
            disable_gate=False,
            disable_fusion_transformer=False,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)
