import torch
import torch.nn as nn

try:
    import deploy._MARLIN as _marlin_cuda
except ImportError:
    _marlin_cuda = None


def _get_marlin_permutations():
    perm = []
    for i in range(32):
        values = []
        col = i // 4
        for block in (0, 1):
            for row in (
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ):
                values.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend(value + 256 * j for value in values)

    perm = torch.tensor(perm, dtype=torch.long)
    interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape(-1, 8)[:, interleave].reshape(-1)

    scale_perm = []
    for i in range(4):
        scale_perm.extend(2 * i + j for j in (0, 1, 8, 9, 16, 17, 24, 25))
    return perm, torch.tensor(scale_perm, dtype=torch.long)


_WEIGHT_PERM, _SCALE_PERM = _get_marlin_permutations()


def is_marlin_available():
    return _marlin_cuda is not None


class LinearW4A16Marlin(nn.Module):
    """Packed INT4 linear using Marlin's FP16 CUDA kernel."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        max_parallel=32,
        output_dtype=None,
    ):
        super().__init__()
        if not is_marlin_available():
            raise RuntimeError(
                "Marlin extension is not built. Run `python setup.py build_ext --inplace`."
            )
        if in_features % 128 != 0:
            raise ValueError(f"in_features={in_features} must be divisible by 128.")
        if out_features % 256 != 0:
            raise ValueError(f"out_features={out_features} must be divisible by 256.")

        self.in_features = in_features
        self.out_features = out_features
        self.max_parallel = max_parallel
        self.output_dtype = output_dtype
        self.register_buffer("weight", torch.empty(0, dtype=torch.int32))
        self.register_buffer("weight_scales", torch.empty(0, dtype=torch.float16))
        self.register_buffer(
            "workspace",
            torch.zeros(out_features // 128 * max_parallel, dtype=torch.int32),
            persistent=False,
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.empty(0, dtype=output_dtype or torch.float16),
            )
        else:
            self.bias = None

    @classmethod
    def from_float(cls, module):
        return cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            output_dtype=module.weight.dtype,
        )

    @torch.no_grad()
    def load_packed_weight(self, packed_weight, scale, device):
        expected_shape = (self.out_features, self.in_features // 2)
        if tuple(packed_weight.shape) != expected_shape:
            raise ValueError(
                f"Packed weight shape {tuple(packed_weight.shape)} does not match {expected_shape}."
            )
        if tuple(scale.shape) not in {(self.out_features,), (self.out_features, 1)}:
            raise ValueError(
                f"Scale shape {tuple(scale.shape)} must be ({self.out_features},) "
                f"or ({self.out_features}, 1)."
            )
        if packed_weight.dtype != torch.uint8:
            raise TypeError(f"Packed weight must be uint8, got {packed_weight.dtype}.")
        if not str(device).startswith("cuda"):
            raise NotImplementedError("LinearW4A16Marlin currently requires CUDA.")

        packed = packed_weight.to(device=device, non_blocking=True)
        quantized = torch.empty(
            (self.out_features, self.in_features),
            dtype=torch.uint8,
            device=device,
        )
        # FlatQuant stores signed two's-complement INT4, low nibble first.
        # Marlin expects unsigned offset-binary values in [0, 15].
        quantized[:, 0::2] = (packed & 0x0F) ^ 0x08
        quantized[:, 1::2] = ((packed >> 4) & 0x0F) ^ 0x08
        del packed

        tile = 16
        quantized = (
            quantized.t()
            .contiguous()
            .reshape(self.in_features // tile, tile, self.out_features // tile, tile)
            .permute(0, 2, 1, 3)
            .reshape(self.in_features // tile, self.out_features * tile)
        )
        perm = _WEIGHT_PERM.to(device=device)
        quantized = quantized.reshape(-1, perm.numel())[:, perm].reshape(quantized.shape)
        quantized = quantized.to(torch.int32)

        marlin_weight = torch.zeros(
            (quantized.shape[0], quantized.shape[1] // 8),
            dtype=torch.int32,
            device=device,
        )
        for index in range(8):
            marlin_weight.bitwise_or_(quantized[:, index::8] << (4 * index))
        self.weight = marlin_weight

        scale = scale.reshape(1, self.out_features).to(
            device=device,
            dtype=torch.float16,
            non_blocking=True,
        )
        scale_perm = _SCALE_PERM.to(device=device)
        self.weight_scales = (
            scale.reshape(-1, scale_perm.numel())[:, scale_perm]
            .reshape(1, self.out_features)
            .contiguous()
        )
        self.workspace = self.workspace.to(device=device)

    def forward(self, x):
        if self.weight.numel() == 0 or self.weight_scales.numel() == 0:
            raise RuntimeError("LinearW4A16Marlin weight has not been loaded.")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input feature dimension {self.in_features}, got {x.shape[-1]}."
            )

        output_dtype = self.output_dtype or x.dtype
        output_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, self.in_features)
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        output = torch.empty(
            (x.shape[0], self.out_features),
            dtype=torch.float16,
            device=x.device,
        )
        _marlin_cuda.mul(
            x,
            self.weight,
            output,
            self.weight_scales,
            self.workspace,
            -1,
            -1,
            -1,
            self.max_parallel,
        )
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        output = output.reshape(output_shape)
        if output.dtype != output_dtype:
            output = output.to(output_dtype)
        return output

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, max_parallel={self.max_parallel}, "
            f"output_dtype={self.output_dtype}"
        )
