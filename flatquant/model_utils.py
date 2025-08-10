import torch
import transformers
import logging
from flatquant.utils import skip
from flatquant.model_tools.llama_utils import apply_flatquant_to_llama
from flatquant.model_tools.llama31_utils import apply_flatquant_to_llama_31


def skip_initialization():
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


def get_llama(model_name, hf_token):
    skip_initialization()
    config = transformers.LlamaConfig.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"
    model = transformers.LlamaForCausalLM.from_pretrained(model_name,
                                                          torch_dtype='auto',
                                                          config=config,
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info(f'---> Loading {model_name} Model with seq_len: {model.seqlen}')
    return model, apply_flatquant_to_llama


def get_llama_31(model_name, hf_token):
    skip_initialization()
    config = transformers.LlamaConfig.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"

    # Multi-GPU를 위한 device_map 설정
    num_layers = config.num_hidden_layers
    layers_per_gpu = num_layers // 2
    
    device_map = {
        "model.embed_tokens": 0,
        "model.norm": 1,
        "lm_head": 1,
    }
    
    for i in range(num_layers):
        if i < layers_per_gpu:
            device_map[f"model.layers.{i}"] = 0
        else:
            device_map[f"model.layers.{i}"] = 1

    model = transformers.LlamaForCausalLM.from_pretrained(model_name,
                                                          torch_dtype='auto',
                                                          config=config,
                                                          use_auth_token=hf_token,
                                                          device_map=device_map,
                                                          offload_folder="offload",
                                                          offload_state_dict=True,
                                                          max_memory={0: "50GB", 1: "50GB"},  # GPU 메모리 제한
                                                          low_cpu_mem_usage=True)
    
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        first_device = model.model.embed_tokens.weight.device

        # 캐시 삭제(있다면)
        for name in ("cos_cached", "sin_cached"):
            if hasattr(model.model.rotary_emb, name):
                setattr(model.model.rotary_emb, name, None)

        # 본체 이동
        model.model.rotary_emb = model.model.rotary_emb.to(first_device)

        # 버퍼도 이동 (inv_freq 등)
        for bname, buf in model.model.rotary_emb.named_buffers(recurse=False):
            if buf is not None and buf.device != first_device:
                model.model.rotary_emb.register_buffer(bname, buf.to(first_device), persistent=False)


    model.seqlen = 2048
    logging.info(f'---> Loading {model_name} Model with seq_len: {model.seqlen}')
    return model, apply_flatquant_to_llama_31


def get_qwen2(model_name, hf_token):
    skip_initialization()
    try:
        from transformers import Qwen2ForCausalLM
    except ImportError:
        logging.error("Qwen2 model is not available in this version of 'transformers'. Please update the library.")
        raise ImportError("Qwen2 model is not available. Ensure you're using a compatible version of the 'transformers' library.")

    config = transformers.Qwen2Config.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"
    model = Qwen2ForCausalLM.from_pretrained(model_name,
                                                          torch_dtype='auto',
                                                          config=config,
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info(f'---> Loading {model_name} Model with seq_len: {model.seqlen}')

    from flatquant.model_tools.qwen_utils import apply_flatquant_to_qwen
    return model, apply_flatquant_to_qwen


def get_opt(model_name):
    skip_initialization()
    model = transformers.OPTForCausalLM.from_pretrained(model_name,
                                                        torch_dtype='auto',
                                                        low_cpu_mem_usage=True)
    model.seqlen = model.config.max_position_embeddings
    logging.info(f'---> Loading {model_name} Model with seq_len: {model.seqlen}')
    raise NotImplementedError("Post-processing for OPT model is not implemented yet.")


# Unified model loading function
def get_model(model_name, hf_token=None):
    if 'llama-3.1' in model_name.lower():
        return get_llama_31(model_name, hf_token)
    elif 'llama-3.3' in model_name:
        return get_llama_31(model_name, hf_token)
    elif 'llama' in model_name:
        return get_llama_31(model_name, hf_token)
    elif 'qwen-2.5' in model_name:
        return get_qwen2(model_name, hf_token)
    else:
        raise ValueError(f'Unknown model {model_name}')

