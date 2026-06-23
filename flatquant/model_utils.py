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
    model = transformers.LlamaForCausalLM.from_pretrained(model_name,
                                                          torch_dtype='auto',
                                                          config=config,
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
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


def get_exaone45(model_name, hf_token):
    skip_initialization()
    try:
        from transformers import Exaone4_5_Config, Exaone4_5_ForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "EXAONE-4.5 requires the nuxlear transformers fork, e.g. "
            "`uv pip install 'git+https://github.com/nuxlear/transformers.git@add-exaone4_5-v5.3.0.dev0'`."
        ) from exc

    config = Exaone4_5_Config.from_pretrained(model_name)
    config._attn_implementation_internal = "eager"
    config.text_config._attn_implementation_internal = "eager"
    config.num_nextn_predict_layers = 0
    config.text_config.num_nextn_predict_layers = 0
    config._num_mtp_layers = 0
    config.text_config._num_mtp_layers = 0

    model_kwargs = {
        "torch_dtype": "auto",
        "config": config,
        "low_cpu_mem_usage": True,
    }
    if hf_token is not None:
        model_kwargs["token"] = hf_token
    try:
        model = Exaone4_5_ForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        if "token" in model_kwargs:
            model_kwargs["use_auth_token"] = model_kwargs.pop("token")
        model = Exaone4_5_ForConditionalGeneration.from_pretrained(model_name, **model_kwargs)

    model.config.num_nextn_predict_layers = 0
    model.config._num_mtp_layers = 0
    model.seqlen = 2048
    logging.info(f'---> Loading {model_name} Model with seq_len: {model.seqlen}')

    from flatquant.model_tools.exaone45_utils import apply_flatquant_to_exaone45
    return model, apply_flatquant_to_exaone45


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
    model_name_lower = model_name.lower()
    if 'exaone-4.5' in model_name_lower or 'exaone4_5' in model_name_lower:
        return get_exaone45(model_name, hf_token)
    if 'llama-3.1' in model_name_lower:
        return get_llama_31(model_name, hf_token)
    elif 'llama' in model_name_lower:
        return get_llama(model_name, hf_token)
    elif 'qwen-2.5' in model_name_lower:
        return get_qwen2(model_name, hf_token)
    else:
        raise ValueError(f'Unknown model {model_name}')

