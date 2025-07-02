import torch
import numpy as np
from typing import Dict, Tuple, Any

class INT4Packer:
    def pack_int4_weight(self, weight: torch.Tensor, sym = False):
        if sym:
            assert weight.min() >= -8 and weight.max() <=7, "not INT4_sym"
        else:
            assert weight.min() >= 0 and weight.max() <=15, "not INT4_asym"

        weight_int8 = weight.to(torch.int8)
        if sym:
            weight_int8 = weight_int8 + 8

        original_shape = weight.shape
        weight_flat = weight_int8.flatten()
        num_elements = weight_flat.numel()

        if num_elements % 2 == 1:
            weight_flat = torch.cat([weight_flat, torch.zeros(1, dtype=torch.int8, device=weight.device)])
        
        weight_pairs = weight_flat.view(-1, 2)
        lower_bits = weight_pairs[:, 0] & 0x0F
        upper_bits = (weight_pairs[:, 1] & 0x0F) << 4
        packed_weight = (upper_bits | lower_bits).to(torch.uint8)

        return packed_weight, original_shape
    

    def unpack_int4_weight(self, packed_weight: torch.Tensor, original_shape: torch.Size, sym = False):
        lower_bits = packed_weight & 0x0F
        upper_bits = (packed_weight >> 4) & 0x0F
        
        unpacked = torch.stack([lower_bits, upper_bits], dim=1).flatten()
        
        if sym:
            unpacked = unpacked - 8
        
        num_elements = np.prod(original_shape)
        unpacked = unpacked[:num_elements]
        
        return unpacked.to(torch.int8).reshape(original_shape)