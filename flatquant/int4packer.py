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
            weight_uint4 = weight_int8 + 8

        original_shape = weight.shape
        weight_flat = weight_uint4.flatten()
        num_elements = weight_flat.numel()

        packed_size = (num_elements + 1) // 2
        packed_weight = torch.zeros(packed_size, dtype = torch.uint8, device = weight.device)

        for i in range(0, num_elements, 2):
            packed_idx = i // 2
            lower_logit = weight_flat[i].item() & 0x0F

            if i + 1 < num_elements:
                upper_logit = (weight_flat[i + 1].item() << 4) & 0xF0
                packed_weight[packed_idx] = upper_logit | lower_logit

        return packed_weight, original_shape
    

    def unpack_int4_weight(self, packed_weight: torch.Tensor, original_shape: torch.Size, sym = False):
        num_elements = np.prod(original_shape)
        unpacked_weight = torch.zeros(num_elements, dtype = torch.int8, device = packed_weight.device)
        
        for i in range(packed_weight.numel()):
            unpacked_weight[i * 2] = (packed_weight[i] & 0x0F)
            if sym:
                unpacked_weight[i * 2] - 8
            
            if i * 2 + 1 < num_elements:
                unpacked_weight[i * 2 + 1] = ((packed_weight[i] >> 4) & 0x0F)
                if sym:
                    unpacked_weight[i * 2 + 1] - 8
        
        return unpacked_weight.reshape(original_shape)