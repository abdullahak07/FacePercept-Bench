from __future__ import annotations


def unwrap_image_features(feat):
    """Normalize transformers image-feature outputs to a tensor."""
    import torch

    if torch.is_tensor(feat):
        return feat
    if getattr(feat, "image_embeds", None) is not None:
        return feat.image_embeds
    if getattr(feat, "pooler_output", None) is not None:
        return feat.pooler_output
    if getattr(feat, "last_hidden_state", None) is not None:
        return feat.last_hidden_state.mean(dim=1)
    if isinstance(feat, (tuple, list)) and feat:
        return feat[0]
    raise TypeError(f"Unsupported encoder output type: {type(feat)}")
