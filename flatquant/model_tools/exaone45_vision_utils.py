"""FlatQuant for the EXAONE-4.5 vision encoder (ViT blocks + patch merger).

The vision transformer block is structurally almost identical to the text decoder
(RMSNorm pre-norms + a gate/up/down SwiGLU MLP + attention), so the FlatQuant idea
carries over directly: an invertible matmul transform is learned on each linear's
input, folded into the weight at reparameterization, and applied online at inference.

The differences handled here versus the text adapter:
  * attention uses a single fused ``qkv`` projection (with bias) and a ``proj`` (with
    bias); there is no KV cache, so only the weight-side transforms are needed.
  * the MLP and merger linears carry biases.
  * the block forward threads ``cu_seqlens`` / vision RoPE / window attention through,
    which is replicated faithfully below while only swapping the linear calls.
"""

import torch
import torch.nn as nn

from flatquant.flat_linear import FlatQuantizedLinear
from flatquant.function_utils import get_decompose_dim
from flatquant.model_tools.model_access import get_vision_module
from flatquant.trans_utils import InvDecomposeTransMatrix, SVDDecomposeTransMatrix

from transformers.models.exaone4_5.modeling_exaone4_5 import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)

try:
    from transformers.models.exaone4_5.modeling_exaone4_5 import is_flash_attention_requested
except ImportError:  # older transformers
    def is_flash_attention_requested(config):
        return getattr(config, "_attn_implementation", "eager") == "flash_attention_2"


def _decompose_trans_cls(args):
    return InvDecomposeTransMatrix if args.direct_inv else SVDDecomposeTransMatrix


def _input_trans(args, in_features):
    """A foldable input transform for one linear, or None when nothing is quantized."""
    if args.w_bits >= 16 and args.a_bits >= 16:
        return None
    left, right = get_decompose_dim(in_features)
    return _decompose_trans_cls(args)(left, right, add_diag=False)


class FlatQuantExaone45VisionMLP(nn.Module):
    def __init__(self, args, module):
        super().__init__()
        self.args = args
        self.act_fn = module.act_fn
        self.up_proj = FlatQuantizedLinear(args, module.up_proj)
        self.gate_proj = FlatQuantizedLinear(args, module.gate_proj)
        self.down_proj = FlatQuantizedLinear(args, module.down_proj)
        self._ori_mode = False
        # gate and up share the same (normed) input, so they share one transform.
        self.up_gate_trans = _input_trans(args, self.up_proj.linear.weight.shape[1])
        self.down_trans = _input_trans(args, self.down_proj.linear.weight.shape[1])

    def _trans_forward(self, x):
        x_ts = self.up_gate_trans(x) if self.up_gate_trans is not None else x
        gate_states = self.gate_proj(x_ts, qa_trans=self.up_gate_trans)
        up_states = self.up_proj(x_ts, qa_trans=self.up_gate_trans)
        x = self.act_fn(gate_states) * up_states
        x_ts = self.down_trans(x) if self.down_trans is not None else x
        return self.down_proj(x_ts, qa_trans=self.down_trans)

    def _ori_forward(self, x):
        x = self.act_fn(self.gate_proj._ori_forward(x)) * self.up_proj._ori_forward(x)
        return self.down_proj._ori_forward(x)

    def forward(self, x):
        return self._ori_forward(x) if self._ori_mode else self._trans_forward(x)

    def reparameterize(self):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()
        self.gate_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.up_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.down_proj.reparameterize(qa_trans=self.down_trans)

    def rep_matrix_only(self):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()


class FlatQuantExaone45VisionAttention(nn.Module):
    def __init__(self, args, module):
        super().__init__()
        self.args = args
        self.config = module.config
        self.dim = module.dim
        self.num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.scaling = module.scaling
        self.attention_dropout = module.attention_dropout
        self.is_causal = module.is_causal
        self.num_key_value_groups = module.num_key_value_groups
        self.q_dim = module.q_dim
        self.kv_dim = module.kv_dim

        self.qkv = FlatQuantizedLinear(args, module.qkv)
        self.proj = FlatQuantizedLinear(args, module.proj)
        self._ori_mode = False

        self.qkv_trans = _input_trans(args, self.qkv.linear.weight.shape[1])
        self.proj_trans = _input_trans(args, self.proj.linear.weight.shape[1])

    def _project_qkv(self, hidden_states, seq_length):
        if self._ori_mode:
            qkv = self.qkv._ori_forward(hidden_states)
        else:
            hidden_ts = self.qkv_trans(hidden_states) if self.qkv_trans is not None else hidden_states
            qkv = self.qkv(hidden_ts, qa_trans=self.qkv_trans)

        if self.num_key_value_groups == 1:
            query_states, key_states, value_states = (
                qkv.reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            )
        else:
            q, kv = torch.split(qkv, [self.q_dim, 2 * self.kv_dim], dim=-1)
            query_states = q.view(seq_length, self.num_heads, self.head_dim)
            kv = kv.view(seq_length, 2, self.num_key_value_groups, self.head_dim)
            key_states = kv[:, 0]
            value_states = kv[:, 1]
            repeat_factor = self.num_heads // self.num_key_value_groups
            key_states = key_states.repeat_interleave(repeat_factor, dim=1)
            value_states = value_states.repeat_interleave(repeat_factor, dim=1)
        return query_states, key_states, value_states

    def _apply_proj(self, attn_output):
        if self._ori_mode:
            return self.proj._ori_forward(attn_output)
        attn_ts = self.proj_trans(attn_output) if self.proj_trans is not None else attn_output
        return self.proj(attn_ts, qa_trans=self.proj_trans)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = self._project_qkv(hidden_states, seq_length)

        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        if is_flash_attention_requested(self.config):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self, query_states, key_states, value_states,
                attention_mask=None, scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen, max_length_k=max_seqlen, is_causal=False, **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    self, q, k, v, attention_mask=None, scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False, **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self._apply_proj(attn_output)

    def reparameterize(self):
        if self.qkv_trans is not None:
            self.qkv_trans.to_eval_mode()
            self.proj_trans.to_eval_mode()
        self.qkv.reparameterize(qa_trans=self.qkv_trans)
        self.proj.reparameterize(qa_trans=self.proj_trans)

    def rep_matrix_only(self):
        if self.qkv_trans is not None:
            self.qkv_trans.to_eval_mode()
            self.proj_trans.to_eval_mode()


class FlatQuantVisionInputLinear(nn.Module):
    """A single foldable input-transformed quantized linear (drop-in nn.Linear).

    Used for the patch merger's ``nn.Sequential`` linears, which each see a distinct
    input and so cannot share a transform with anything else.
    """

    def __init__(self, args, linear):
        super().__init__()
        self.args = args
        self.proj = FlatQuantizedLinear(args, linear)
        self.trans = _input_trans(args, self.proj.linear.weight.shape[1])
        self._ori_mode = False

    def forward(self, x):
        if self._ori_mode:
            return self.proj._ori_forward(x)
        x_ts = self.trans(x) if self.trans is not None else x
        return self.proj(x_ts, qa_trans=self.trans)

    def reparameterize(self):
        if self.trans is not None:
            self.trans.to_eval_mode()
        self.proj.reparameterize(qa_trans=self.trans)

    def rep_matrix_only(self):
        if self.trans is not None:
            self.trans.to_eval_mode()


def iter_vision_flat_modules(model):
    """Yield every FlatQuant vision wrapper (for calibration / reparameterization)."""
    visual = get_vision_module(model)
    if visual is None:
        return
    for block in visual.blocks:
        yield block.attn
        yield block.mlp
    merger_mlp = visual.merger.mlp
    for idx in range(len(merger_mlp)):
        if isinstance(merger_mlp[idx], FlatQuantVisionInputLinear):
            yield merger_mlp[idx]


def apply_flatquant_to_exaone45_vision(args, model):
    """Wrap the vision encoder's attention, MLP and merger linears with FlatQuant."""
    visual = get_vision_module(model)
    if visual is None:
        return model
    for block in visual.blocks:
        block.attn = FlatQuantExaone45VisionAttention(args, block.attn)
        block.mlp = FlatQuantExaone45VisionMLP(args, block.mlp)
    merger_mlp = visual.merger.mlp
    for idx in range(len(merger_mlp)):
        if isinstance(merger_mlp[idx], nn.Linear):
            merger_mlp[idx] = FlatQuantVisionInputLinear(args, merger_mlp[idx])
    return model


def reparameterize_vision(model):
    for wrapper in iter_vision_flat_modules(model):
        wrapper.reparameterize()
    return model
