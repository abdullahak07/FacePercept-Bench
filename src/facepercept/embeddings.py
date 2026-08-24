def unwrap_image_features(feat):
    import torch

    if torch.is_tensor(feat):
        return feat

    for name in ("image_embeds", "pooler_output"):
        value = getattr(feat, name, None)
        if value is not None:
            return value

    hidden = getattr(feat, "last_hidden_state", None)
    if hidden is not None:
        return hidden.mean(dim=1)

    if isinstance(feat, (tuple, list)) and len(feat):
        return feat[0]

    raise TypeError(f"cannot use encoder output {type(feat)}")
