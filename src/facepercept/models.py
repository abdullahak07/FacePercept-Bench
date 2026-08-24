def load_qwen25_vl(model_id, dtype=None, device="cpu"):
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    return model.to(device)
