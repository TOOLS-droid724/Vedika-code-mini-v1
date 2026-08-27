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
"""PyTorch Vedika-Code-Pro-v1 model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
from functools import lru_cache

import torch
from torch import nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_vedika_code_pro_v1 import VedikaCodeProV1Config
except ImportError:
    from configuration_vedika_code_pro_v1 import VedikaCodeProV1Config


def _get_context_manager():
    try:
        from contextlib import nullcontext
        return nullcontext
    except ImportError:
        from contextlib import contextmanager
        @contextmanager
        def nullcontext(enter_result=None):
            yield enter_result
        return nullcontext


@lru_cache(2)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int
) -> torch.Tensor:
    """Precomputes complex exponentials for rotary embeddings with YaRN scaling."""
    
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Applies rotary positional embeddings in-place."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):
    """Compresses KV cache via learned gated pooling."""
    
    def __init__(self, config: VedikaCodeProV1Config, compress_ratio: int = 4, head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - self.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        self.wkv = nn.Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.wgate = nn.Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.kv_cache: Optional[torch.Tensor] = None
        self.register_buffer("kv_state", torch.zeros(config.max_batch_size, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("score_state", torch.full((config.max_batch_size, coff * compress_ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32), persistent=False)
        self.freqs_cis: Optional[torch.Tensor] = None

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio:cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio:cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset:offset + remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset:offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention."""
    
    def __init__(self, config: VedikaCodeProV1Config, compress_ratio: int = 4):
        super().__init__()
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio

        self.compressor = Compressor(config, compress_ratio, self.head_dim, True)
        self.register_buffer("kv_cache", torch.zeros(config.max_batch_size, config.max_position_embeddings // compress_ratio, self.head_dim), persistent=False)
        self.freqs_cis = None

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        fp4_block_size = 32
        # Simulate FP4 quantization
        q = q / (q.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5))
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs


class Attention(nn.Module):
    """Multi-head Latent Attention (MLA) with sliding window + optional KV compression."""
    
    def __init__(self, layer_id: int, config: VedikaCodeProV1Config):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.window_size = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_id] if layer_id < len(config.compress_ratios) else 0
        self.eps = config.rms_norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = nn.Linear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * config.o_lora_rank, bias=False, dtype=torch.bfloat16)
        self.wo_b = nn.Linear(self.n_groups * config.o_lora_rank, self.dim, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(config, self.compress_ratio)
            else:
                self.indexer = None

        kv_cache_size = self.window_size + (config.max_position_embeddings // self.compress_ratio if self.compress_ratio else 0)
        self.register_buffer("kv_cache", torch.zeros(config.max_batch_size, kv_cache_size, self.head_dim), persistent=False)
        if self.compress_ratio:
            original_seq_len = config.rope_scaling.get("original_max_position_embeddings", 0)
            rope_theta = config.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, config.rope_theta
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim, config.max_position_embeddings, original_seq_len,
            rope_theta, config.rope_scaling.get("factor", 1),
            config.rope_scaling.get("beta_fast", 32), config.rope_scaling.get("beta_slow", 1)
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        # q
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        # win kv & topk_idxs
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # compress kv & attn
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = self._sparse_attn(q, kv, topk_idxs)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = self._sparse_attn(q, self.kv_cache[:bsz], topk_idxs)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # o
        o = o.view(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x

    def _sparse_attn(self, q, kv, topk_idxs, softmax_scale=None):
        """Simplified sparse attention without custom kernels."""
        if softmax_scale is None:
            softmax_scale = self.softmax_scale
        bsz, seqlen, n_heads, head_dim = q.size()
        kv_seqlen = kv.size(1)
        
        # Gather kv values at topk indices
        topk_idxs_flat = topk_idxs.view(bsz, seqlen, -1)
        valid_mask = topk_idxs_flat >= 0
        
        # Expand q to match kv dims for gathering
        q_expanded = q.view(bsz, seqlen, n_heads, 1, head_dim)
        
        # Simple dense attention fallback for compatibility
        attn_weights = torch.einsum("bshd,bthd->bsht", q, kv) * softmax_scale
        
        # Apply sparse mask based on topk_idxs
        mask = torch.full((bsz, seqlen, n_heads, kv_seqlen), float("-inf"), device=q.device, dtype=q.dtype)
        for i in range(topk_idxs_flat.size(-1)):
            idxs = topk_idxs_flat[:, :, i]
            valid = idxs >= 0
            batch_idx = torch.arange(bsz, device=q.device).view(-1, 1, 1)
            seq_idx = torch.arange(seqlen, device=q.device).view(1, -1, 1)
            head_idx = torch.arange(n_heads, device=q.device).view(1, 1, -1)
            mask[batch_idx, seq_idx, head_idx, idxs] = torch.where(valid, torch.tensor(0.0, device=q.device), torch.tensor(float("-inf"), device=q.device))
        
        attn_weights = attn_weights + mask
        attn_probs = attn_weights.softmax(dim=-1)
        o = torch.einsum("bsht,bthd->bshd", attn_probs, kv)
        return o


class Gate(nn.Module):
    """MoE gating: computes expert routing scores and selects top-k experts."""
    
    def __init__(self, layer_id: int, config: VedikaCodeProV1Config):
        super().__init__()
        self.dim = config.hidden_size
        self.topk = config.num_experts_per_tok
        self.score_func = config.scoring_func
        self.route_scale = config.routed_scaling_factor
        self.hash = layer_id < config.num_hash_layers
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        if self.hash:
            self.tid2eid = nn.Parameter(torch.empty(config.vocab_size, config.num_experts_per_tok, dtype=torch.int32), requires_grad=False)
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN."""
    
    def __init__(self, dim: int, inter_dim: int, swiglu_limit=0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """Mixture-of-Experts: gate routes each token to top-k routed experts + 1 shared expert."""
    
    def __init__(self, layer_id: int, config: VedikaCodeProV1Config):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.gate = Gate(layer_id, config)
        self.experts = nn.ModuleList([
            Expert(config.hidden_size, config.moe_intermediate_size, swiglu_limit=config.swiglu_limit)
            for _ in range(self.n_routed_experts)
        ])
        assert config.n_shared_experts == 1
        self.shared_experts = Expert(config.hidden_size, config.moe_intermediate_size, swiglu_limit=config.swiglu_limit)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


def hc_split_sinkhorn(mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    """Split mixes into pre, post, comb components using Sinkhorn iterations."""
    mix_hc = mixes.size(-1)
    pre_mix = mix_hc // 3
    post_mix = mix_hc // 3
    comb_mix = mix_hc - pre_mix - post_mix
    
    # Simplified HC mixing without full Sinkhorn
    pre = torch.sigmoid(mixes[..., :pre_mix] * scale[..., :1, None] + base[..., :pre_mix]) + eps
    post = torch.sigmoid(mixes[..., pre_mix:pre_mix + post_mix] * scale[..., 1:2, None] + base[..., pre_mix:pre_mix + post_mix]) + eps
    comb = torch.sigmoid(mixes[..., pre_mix + post_mix:] * scale[..., 2:3, None] + base[..., pre_mix + post_mix:]) + eps
    comb = comb.view(*comb.shape[:-1], hc_mult, hc_mult)
    
    return pre, post, comb


class Block(nn.Module):
    """Transformer block with Hyper-Connections (HC) mixing."""
    
    def __init__(self, layer_id: int, config: VedikaCodeProV1Config):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = config.rms_norm_eps
        self.attn = Attention(layer_id, config)
        self.ffn = MoE(layer_id, config)
        self.attn_norm = RMSNorm(config.hidden_size, self.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, self.norm_eps)
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class MTPBlock(Block):
    """Multi-Token Prediction block."""
    
    def __init__(self, layer_id: int, config: VedikaCodeProV1Config):
        super().__init__(layer_id, config)
        self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.enorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))
        self.embed: Optional[nn.Embedding] = None
        self.head: Optional[nn.Linear] = None

    def hc_head(self, x: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed is not None and self.head is not None
        e = self.embed(input_ids)
        e = self.enorm(e)
        x = self.hnorm(x)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids)
        x = self.hc_head(x)
        logits = F.linear(self.norm(x).float(), self.head.weight)
        return logits


class VedikaCodeProV1ForCausalLM(PreTrainedModel):
    """Vedika-Code-Pro-v1 model for causal language modeling."""
    
    config_class = VedikaCodeProV1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Block", "MTPBlock"]
    
    def __init__(self, config: VedikaCodeProV1Config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.n_mtp_layers = config.num_nextn_predict_layers
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.mtp = nn.ModuleList()
        for layer_id in range(self.n_mtp_layers):
            mtp_block = MTPBlock(config.num_hidden_layers + layer_id, config)
            mtp_block.embed = self.embed_tokens
            mtp_block.head = self.lm_head
            self.mtp.append(mtp_block)
        
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(hc_dim, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(hc_dim))
        self.hc_head_scale = nn.Parameter(torch.empty(1))
        
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...] | CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Expand to hc_mult copies for Hyper-Connections
        hidden_states = inputs_embeds.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        
        # Determine start_pos for incremental decoding
        start_pos = 0
        if past_key_values is not None:
            start_pos = past_key_values[0][0].size(1) if isinstance(past_key_values[0], tuple) else 0
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, start_pos, input_ids if input_ids is not None else inputs_embeds.argmax(-1))
        
        # Apply HC head for final logits
        shape, dtype = hidden_states.size(), hidden_states.dtype
        hidden_states_flat = hidden_states.flatten(2).float()
        rsqrt = torch.rsqrt(hidden_states_flat.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
        mixes = F.linear(hidden_states_flat, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.config.hc_eps
        hidden_states = torch.sum(pre.unsqueeze(-1) * hidden_states_flat.view(shape), dim=2).to(dtype)
        
        hidden_states = self.norm(hidden_states)
        # Take last token for logits
        logits = self.lm_head(hidden_states[:, -1])
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_layer_past = tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            reordered_past += (reordered_layer_past,)
        return reordered_past
