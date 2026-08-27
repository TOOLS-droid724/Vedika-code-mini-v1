# coding=utf-8
# Copyright 2025 Veda Labs. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Vedika-Code-Pro-v1 model configuration."""

from transformers import PretrainedConfig


class VedikaCodeProV1Config(PretrainedConfig):
    """Configuration class for Vedika-Code-Pro-v1 models."""
    
    model_type = "vedika_code_pro_v1"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        vocab_size=129280,
        hidden_size=7168,
        moe_intermediate_size=3072,
        num_hidden_layers=61,
        num_hash_layers=3,
        num_attention_heads=128,
        num_key_value_heads=1,
        n_routed_experts=384,
        n_shared_experts=1,
        num_experts_per_tok=6,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=2.5,
        swiglu_limit=10.0,
        q_lora_rank=1536,
        head_dim=512,
        qk_rope_head_dim=64,
        o_groups=16,
        o_lora_rank=1024,
        sliding_window=128,
        rope_theta=10000.0,
        rope_scaling=None,
        compress_rope_theta=160000.0,
        compress_ratios=None,
        rms_norm_eps=1e-6,
        max_batch_size=4,
        max_position_embeddings=1048576,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=1024,
        attention_dropout=0.0,
        initializer_range=0.02,
        tie_word_embeddings=False,
        bos_token_id=0,
        eos_token_id=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hash_layers = num_hash_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.q_lora_rank = q_lora_rank
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling if rope_scaling is not None else {}
        self.compress_rope_theta = compress_rope_theta
        self.compress_ratios = compress_ratios if compress_ratios is not None else []
        self.rms_norm_eps = rms_norm_eps
        self.max_batch_size = max_batch_size
        self.max_position_embeddings = max_position_embeddings
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings
        self.num_nextn_predict_layers = kwargs.get("num_nextn_predict_layers", 1)
        
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
