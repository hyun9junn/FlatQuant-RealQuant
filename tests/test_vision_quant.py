"""Tests for vision-encoder weight quantization (RTN W4) and the W4A16 runtime glue.

These cover the EXAONE-4.5 vision path: ``rtn_quantize_vision`` must find and quantize
every vision ``nn.Linear`` (and skip the patch_embed Conv3d), emit checkpoint-matching
quantizer keys, and the packed weights must round-trip through ``LinearW4A16`` to match
the RTN-dequantized reference (bias included).
"""

import os
import tempfile
import types
import unittest

import torch
import torch.nn as nn
from safetensors.torch import save_file

from deploy.nn import LinearW4A16, is_marlin_available
from deploy.transformers import vision_quant
from flatquant.flat_utils import _pack_i4
from gptq_utils import rtn_quantize_vision


def _vision_args(w_bits=4):
    return types.SimpleNamespace(
        quantize_vision=True, w_bits=w_bits, w_asym=False, gptq_mse=False
    )


class _Attn(nn.Module):
    def __init__(self, dim, qkv_out):
        super().__init__()
        self.qkv = nn.Linear(dim, qkv_out, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)


class _MLP(nn.Module):
    def __init__(self, dim, inter):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter, bias=True)
        self.up_proj = nn.Linear(dim, inter, bias=True)
        self.down_proj = nn.Linear(inter, dim, bias=True)


class _Block(nn.Module):
    def __init__(self, dim, inter, qkv_out):
        super().__init__()
        self.attn = _Attn(dim, qkv_out)
        self.mlp = _MLP(dim, inter)


class _Merger(nn.Module):
    def __init__(self, mdim, out):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(mdim, mdim), nn.GELU(), nn.Linear(mdim, out))


class _PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Conv3d, like the real patch_embed -- must be left in fp16, never quantized.
        self.proj = nn.Conv3d(3, dim, kernel_size=(2, 14, 14))


class _Visual(nn.Module):
    def __init__(self, dim=256, inter=512, qkv_out=384, mdim=1024, out=256):
        super().__init__()
        self.patch_embed = _PatchEmbed(dim)
        self.blocks = nn.ModuleList([_Block(dim, inter, qkv_out)])
        self.merger = _Merger(mdim, out)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Visual()


class _Model(nn.Module):
    """Stand-in for Exaone4_5_ForConditionalGeneration: vision lives at model.model.visual."""

    def __init__(self):
        super().__init__()
        self.model = _Inner()


EXPECTED_KEYS = {
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.mlp.gate_proj",
    "model.visual.blocks.0.mlp.up_proj",
    "model.visual.blocks.0.mlp.down_proj",
    "model.visual.merger.mlp.0",
    "model.visual.merger.mlp.2",
}


class RtnQuantizeVisionTest(unittest.TestCase):
    def test_quantizes_every_linear_and_skips_conv(self):
        model = _Model().eval()
        quantizers = {}
        rtn_quantize_vision(model, "cpu", _vision_args(), quantizers)

        self.assertEqual(set(quantizers.keys()), EXPECTED_KEYS)
        # The Conv3d patch_embed must not be quantized.
        self.assertNotIn("model.visual.patch_embed.proj", quantizers)

    def test_w_bits_16_is_noop(self):
        model = _Model().eval()
        quantizers = {}
        rtn_quantize_vision(model, "cpu", _vision_args(w_bits=16), quantizers)
        self.assertEqual(quantizers, {})


class VisionPackLoadRoundTripTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_packed_vision_linear_matches_dequant_reference(self):
        torch.manual_seed(3)
        model = _Model().eval()
        # Capture a fresh-weight reference Linear (with bias) before RTN mutates it.
        block = model.model.visual.blocks[0]
        orig_bias = block.attn.qkv.bias.detach().clone()

        quantizers = {}
        rtn_quantize_vision(model, "cuda", _vision_args(), quantizers)

        # After RTN, the module weight holds the dequantized reference.
        ref_linear = block.attn.qkv.to("cuda")
        ref_weight = ref_linear.weight.data.clone()

        quantizer = quantizers["model.visual.blocks.0.attn.qkv"]
        scale = quantizer.scale.to("cuda")
        maxq = quantizer.maxq.to("cuda")
        # Reproduce the safetensors save path: re-quantize the dequant weight and pack.
        param_quant = torch.clamp((ref_weight / scale).round(), -(maxq + 1), maxq)
        packed = _pack_i4(param_quant.to(torch.int8)).contiguous().cpu()

        linear = LinearW4A16.from_float(ref_linear).cuda()
        linear.load_packed_weight(packed, scale.cpu(), "cuda")
        linear.bias = orig_bias.to("cuda", dtype=torch.bfloat16)

        x = torch.randn(8, ref_linear.in_features, device="cuda", dtype=torch.bfloat16)
        output = linear(x)
        reference = x.float() @ ref_weight.float().T + orig_bias.cuda().float()

        self.assertEqual(output.shape, (8, ref_linear.out_features))
        self.assertLess((output.float() - reference).abs().mean().item(), 0.05)


def _write_vision_checkpoint(model, quantizers, directory):
    """Serialize RTN-quantized vision linears the way the safetensors saver does."""
    state_dict = {}
    for name, quantizer in quantizers.items():
        linear = model.get_submodule(name)
        scale = quantizer.scale
        maxq = quantizer.maxq
        param_quant = torch.clamp((linear.weight.data / scale).round(), -(maxq + 1), maxq)
        state_dict[f"{name}.weight"] = _pack_i4(param_quant.to(torch.int8)).contiguous().cpu()
        state_dict[f"{name}.bias"] = linear.bias.data.to(torch.float16).cpu()
        state_dict[f"quantizer.{name}.scale"] = scale.contiguous().cpu()
    save_file(state_dict, os.path.join(directory, "model.safetensors"))


class DeployVisionLoadTest(unittest.TestCase):
    def test_select_prefers_marlin_when_available(self):
        from deploy.nn import LinearW4A16Marlin

        cls = vision_quant.select_vision_linear_cls("cuda", prefer_marlin=True)
        if torch.cuda.is_available() and is_marlin_available():
            major, _ = torch.cuda.get_device_capability("cuda")
            expected = LinearW4A16Marlin if major >= 8 else LinearW4A16
            self.assertIs(cls, expected)
        # int4pack fallback must be chosen when Marlin is explicitly disabled.
        self.assertIs(vision_quant.select_vision_linear_cls("cuda", prefer_marlin=False), LinearW4A16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_replace_and_load_matches_reference(self):
        torch.manual_seed(5)
        model = _Model().eval()
        quantizers = {}
        rtn_quantize_vision(model, "cuda", _vision_args(), quantizers)
        model = model.cpu()

        with tempfile.TemporaryDirectory() as tmp:
            _write_vision_checkpoint(model, quantizers, tmp)

            fresh = _Model().eval()
            linear_cls = LinearW4A16  # int4pack: deterministic, no Marlin build needed
            replaced = vision_quant.replace_vision_linears(fresh, linear_cls)
            self.assertEqual(replaced, len(EXPECTED_KEYS))

            vision_quant.get_vision_module(fresh).to("cuda")
            loaded = vision_quant.load_vision_packed_weights(fresh, tmp, linear_cls, "cuda")
            self.assertEqual(loaded, len(EXPECTED_KEYS))

            # Every loaded vision linear should reproduce the RTN-dequantized reference.
            ref_block = model.model.visual.blocks[0].to("cuda")
            new_block = fresh.model.visual.blocks[0]
            x = torch.randn(8, ref_block.attn.qkv.in_features, device="cuda", dtype=torch.float32)
            ref = ref_block.attn.qkv(x)
            out = new_block.attn.qkv(x)
            rel_err = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
            self.assertLess(rel_err.item(), 0.02)


if __name__ == "__main__":
    unittest.main()
