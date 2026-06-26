"""Numerical correctness tests for EXAONE-4.5 vision FlatQuant wrappers.

The FlatQuant transforms are invertible by construction, so wrapping a vision block
and folding the transforms back must reproduce the original block exactly (modulo
weight quantization). These tests pin that property -- in particular they guard the
hand-replicated vision attention forward (fused qkv split, vision RoPE, cu_seqlens
chunked attention) against drift from the upstream modeling code.
"""

import os
import types
import unittest

import torch

from flatquant.model_tools.exaone45_vision_calib import cali_flat_quant_vision
from flatquant.model_tools.exaone45_vision_utils import (
    FlatQuantExaone45VisionAttention,
    FlatQuantExaone45VisionMLP,
    FlatQuantVisionInputLinear,
    apply_flatquant_to_exaone45_vision,
    iter_vision_flat_modules,
    reparameterize_vision,
)
from flatquant.quant_utils import set_quantizer_state

from transformers.models.exaone4_5.configuration_exaone4_5 import Exaone4_5_VisionConfig
from transformers.models.exaone4_5.modeling_exaone4_5 import (
    Exaone4_5_PatchMerger,
    Exaone4_5_VisionBlock,
    Exaone4_5_VisionPreTrainedModel,
)


def _vision_config():
    # num_key_value_heads=1 keeps the attention on the non-GQA path, which avoids an
    # unrelated repeat_kv double-count in the upstream eager kernel for tiny shapes.
    cfg = Exaone4_5_VisionConfig(
        hidden_size=64,
        num_heads=4,
        num_key_value_heads=1,
        intermediate_size=128,
        depth=2,
        out_hidden_size=64,
        spatial_merge_size=2,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _flat_args(w_bits=16, a_bits=16):
    return types.SimpleNamespace(
        w_bits=w_bits, a_bits=a_bits, w_asym=False, a_asym=False,
        lac=False, a_groupsize=-1, lwc=False, direct_inv=False, gptq_mse=False,
    )


def _block_inputs(cfg, seq=8, dtype=torch.float64):
    head_dim = cfg.hidden_size // cfg.num_heads
    x = torch.randn(seq, cfg.hidden_size, dtype=dtype)
    cu_seqlens = torch.tensor([0, seq], dtype=torch.int32)
    rotary = torch.randn(seq, head_dim // 2, dtype=dtype)
    return x, cu_seqlens, rotary


class VisionFlatQuantBlockTest(unittest.TestCase):
    def test_w16a16_wrapper_reproduces_block_exactly(self):
        torch.manual_seed(0)
        cfg = _vision_config()
        block = Exaone4_5_VisionBlock(cfg).eval().double()
        x, cu, rot = _block_inputs(cfg)
        with torch.no_grad():
            ref = block(x, cu_seqlens=cu, rotary_pos_emb=rot)

        args = _flat_args(w_bits=16, a_bits=16)  # transforms are no-ops, quant is identity
        block.attn = FlatQuantExaone45VisionAttention(args, block.attn).double()
        block.mlp = FlatQuantExaone45VisionMLP(args, block.mlp).double()
        with torch.no_grad():
            out = block(x, cu_seqlens=cu, rotary_pos_emb=rot)
        self.assertLess((ref - out).abs().max().item(), 1e-10)

    def test_transforms_are_invertible_through_wrapper(self):
        torch.manual_seed(1)
        cfg = _vision_config()
        block = Exaone4_5_VisionBlock(cfg).eval().double()
        x, cu, rot = _block_inputs(cfg)
        with torch.no_grad():
            ref = block(x, cu_seqlens=cu, rotary_pos_emb=rot)

        args = _flat_args(w_bits=4, a_bits=16)  # transforms now exist
        block.attn = FlatQuantExaone45VisionAttention(args, block.attn).double()
        block.mlp = FlatQuantExaone45VisionMLP(args, block.mlp).double()
        # Disable weight quantization so only the transform fold is exercised.
        set_quantizer_state(block, enable=False)

        with torch.no_grad():
            train_path = block(x, cu_seqlens=cu, rotary_pos_emb=rot)
        self.assertLess((ref - train_path).abs().max().item(), 1e-10)

        block.attn.reparameterize()
        block.mlp.reparameterize()
        with torch.no_grad():
            eval_path = block(x, cu_seqlens=cu, rotary_pos_emb=rot)
        self.assertLess((ref - eval_path).abs().max().item(), 1e-10)


class _Inner(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.visual = torch.nn.Module()
        self.visual.blocks = torch.nn.ModuleList(
            [Exaone4_5_VisionBlock(cfg) for _ in range(cfg.depth)]
        )
        self.visual.merger = Exaone4_5_PatchMerger(
            dim=cfg.out_hidden_size, context_dim=cfg.hidden_size,
            spatial_merge_size=cfg.spatial_merge_size,
        )


class _Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = _Inner(cfg)


class VisionFlatQuantApplyTest(unittest.TestCase):
    def test_apply_wraps_blocks_and_merger(self):
        cfg = _vision_config()
        model = _Model(cfg)
        apply_flatquant_to_exaone45_vision(_flat_args(w_bits=4), model)

        for block in model.model.visual.blocks:
            self.assertIsInstance(block.attn, FlatQuantExaone45VisionAttention)
            self.assertIsInstance(block.mlp, FlatQuantExaone45VisionMLP)
        merger_mlp = model.model.visual.merger.mlp
        self.assertIsInstance(merger_mlp[0], FlatQuantVisionInputLinear)
        self.assertIsInstance(merger_mlp[2], FlatQuantVisionInputLinear)
        self.assertIsInstance(merger_mlp[1], torch.nn.GELU)

        # 2 blocks * (attn + mlp) + 2 merger linears = 6 FlatQuant wrappers.
        self.assertEqual(len(list(iter_vision_flat_modules(model))), 6)
        # reparameterize over every wrapper must succeed.
        reparameterize_vision(model)


def _tower_config():
    cfg = Exaone4_5_VisionConfig(
        hidden_size=64, num_heads=4, num_key_value_heads=1, intermediate_size=128,
        depth=2, out_hidden_size=64, spatial_merge_size=2, patch_size=2,
        temporal_patch_size=1, in_channels=3, window_size=4, fullatt_block_indexes=[1],
    )
    cfg._attn_implementation = "eager"
    return cfg


class _TowerModel(torch.nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.model = types.SimpleNamespace(visual=tower)


class VisionFlatQuantCalibrationTest(unittest.TestCase):
    def test_calibration_runs_and_reparameterizes(self):
        torch.manual_seed(0)
        cfg = _tower_config()
        tower = Exaone4_5_VisionPreTrainedModel(cfg).eval()
        model = _TowerModel(tower)
        args = _flat_args(w_bits=4, a_bits=16)
        args.cali_trans = True
        args.flat_lr = 5e-3
        args.epochs = 2
        apply_flatquant_to_exaone45_vision(args, model)

        in_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        samples = [(torch.randn(16, in_dim), torch.tensor([[1, 4, 4]])) for _ in range(3)]

        flat_params = cali_flat_quant_vision(args, model, samples, "cpu", logger=None)
        # Every block (attn + mlp) and both merger linears should be calibrated.
        self.assertEqual(
            set(flat_params.keys()),
            {"blocks.0.attn", "blocks.0.mlp", "blocks.1.attn", "blocks.1.mlp",
             "merger.mlp.0", "merger.mlp.2"},
        )
        # The folded eval path must still produce a valid forward.
        reparameterize_vision(model)
        with torch.no_grad():
            out = tower(samples[0][0], samples[0][1])
        self.assertEqual(out.last_hidden_state.shape, (16, cfg.hidden_size))


class _InnerReal(torch.nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.visual = tower


class _RealModel(torch.nn.Module):
    """nn.Module container so get_submodule / named_parameters resolve model.visual.*"""

    def __init__(self, tower):
        super().__init__()
        self.model = _InnerReal(tower)


def _write_flatquant_vision_checkpoint(model, quantizers, directory):
    from flatquant.flat_utils import _pack_i4

    state = {}
    for name, param in model.named_parameters():
        layer = name.rsplit(".", 1)[0] if (name.endswith(".weight") or name.endswith(".bias")) else name
        if layer in quantizers and name.endswith(".weight"):
            q = quantizers[layer]
            scale, maxq = q.scale.to(param.device), q.maxq.to(param.device)
            pq = torch.clamp((param.data / scale).round(), -(maxq + 1), maxq)
            state[name] = _pack_i4(pq.to(torch.int8)).contiguous().cpu()
        else:
            state[name] = param.data.to(torch.float16).cpu()
    for layer, q in quantizers.items():
        state[f"quantizer.{layer}.scale"] = q.scale.cpu()
    from safetensors.torch import save_file

    save_file(state, os.path.join(directory, "model.safetensors"))


class DeployVisionFlatQuantTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deploy_load_matches_reparameterized_reference(self):
        import tempfile

        from deploy.transformers import vision_quant
        from flatquant.model_tools.exaone45_vision_utils import apply_flatquant_to_exaone45_vision
        from gptq_utils import rtn_quantize_vision

        torch.manual_seed(0)
        # Dims chosen so every vision linear satisfies both packed kernels
        # (in_features % 128 == 0, out_features % 256 == 0 for Marlin).
        cfg = Exaone4_5_VisionConfig(
            hidden_size=256, num_heads=4, num_key_value_heads=1, intermediate_size=512,
            depth=2, out_hidden_size=256, spatial_merge_size=2, patch_size=2,
            temporal_patch_size=1, in_channels=3, window_size=4, fullatt_block_indexes=[1],
        )
        cfg._attn_implementation = "eager"
        args = _flat_args(w_bits=4, a_bits=16)

        # Build, FlatQuant-wrap, fold transforms, then RTN-quantize the vision tower.
        ref_model = _RealModel(Exaone4_5_VisionPreTrainedModel(cfg).eval())
        apply_flatquant_to_exaone45_vision(args, ref_model)
        reparameterize_vision(ref_model)
        quantizers = {}
        rtn_quantize_vision(ref_model, "cuda", args, quantizers)

        in_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        pixel_values = torch.randn(16, in_dim)
        grid_thw = torch.tensor([[1, 4, 4]])

        ref_model.model.visual.to("cuda")
        with torch.no_grad():
            reference = ref_model.model.visual(pixel_values.cuda(), grid_thw.cuda()).last_hidden_state

        with tempfile.TemporaryDirectory() as tmp:
            _write_flatquant_vision_checkpoint(ref_model, quantizers, tmp)

            fresh = _RealModel(Exaone4_5_VisionPreTrainedModel(cfg).eval())
            vision_args = types.SimpleNamespace(
                w_bits=4, a_bits=16, w_asym=False, a_asym=False, lac=False,
                a_groupsize=-1, lwc=False, direct_inv=False, cali_trans=True,
            )
            vision_quant.load_vision_flatquant(fresh, tmp, vision_args, "cuda")
            with torch.no_grad():
                out = fresh.model.visual(pixel_values.cuda(), grid_thw.cuda()).last_hidden_state

        rel_err = (out.float() - reference.float()).abs().mean() / reference.float().abs().mean()
        self.assertLess(rel_err.item(), 0.05)


if __name__ == "__main__":
    unittest.main()
