from __future__ import annotations


def load_qwen25_vl(model_id: str, dtype=None, device: str = "cpu"):
    """Load Qwen2.5-VL with the correct multimodal generation class."""
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    return model.to(device)
