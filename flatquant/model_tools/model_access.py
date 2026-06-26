def get_transformer_backbone(model):
    """Return the decoder-only text backbone used by FlatQuant calibration."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    if hasattr(model, "layers"):
        return model
    raise AttributeError("Could not find transformer decoder layers on this model.")


def get_transformer_layers(model):
    return get_transformer_backbone(model).layers


def get_transformer_config(model):
    backbone = get_transformer_backbone(model)
    return getattr(backbone, "config", model.config)


def get_transformer_layer_prefix(model):
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return "model.layers"
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return "model.language_model.layers"
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return "language_model.layers"
    if hasattr(model, "layers"):
        return "layers"
    return "model.layers"


def is_exaone45_model(model):
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    return model_type == "exaone4_5"


def get_vision_module(model):
    """Return the vision encoder (tower + merger) if the model has one, else None.

    EXAONE-4.5 wraps its ViT under ``model.model.visual``; other layouts expose it
    directly as ``model.visual``. The vision tower is optional, so callers should
    treat ``None`` as "text-only model, nothing to quantize here".
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "visual"):
        return inner.visual
    if hasattr(model, "visual"):
        return model.visual
    return None


def first_hidden_state(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output
