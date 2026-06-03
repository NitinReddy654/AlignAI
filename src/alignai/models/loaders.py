"""Model loading utilities for base, FFT, LoRA, and QLoRA variants."""

from __future__ import annotations

from typing import Any, Tuple

from alignai.hardware import detect_hardware, get_torch_dtype
from alignai.logging_utils import setup_logging

logger = setup_logging(__name__)


def _default_bnb_config() -> Any:
    from transformers import BitsAndBytesConfig

    hw = detect_hardware()
    compute_dtype = get_torch_dtype(hw.dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def _ensure_pad_token(tokenizer: Any, model: Any) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


def load_base_model(model_id: str, device: str | None = None) -> Tuple[Any, Any]:
    """Load base Qwen model for inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hw = detect_hardware()
    device = device or hw.device
    dtype = get_torch_dtype(hw.dtype)

    logger.info("Loading base model: %s on %s", model_id, device)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _ensure_pad_token(tokenizer, model)
    return model, tokenizer


def load_fft_model(checkpoint_path: str, device: str | None = None) -> Tuple[Any, Any]:
    """Load fully fine-tuned model checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hw = detect_hardware()
    device = device or hw.device
    dtype = get_torch_dtype(hw.dtype)

    logger.info("Loading FFT checkpoint: %s", checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=dtype, trust_remote_code=True)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    _ensure_pad_token(tokenizer, model)
    return model, tokenizer


def load_lora_model(
    base_model_id: str,
    adapter_path: str,
    merge: bool = True,
    device: str | None = None,
) -> Tuple[Any, Any]:
    """Load LoRA adapter on base model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hw = detect_hardware()
    device = device or hw.device
    dtype = get_torch_dtype(hw.dtype)

    logger.info("Loading LoRA: base=%s adapter=%s", base_model_id, adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model = PeftModel.from_pretrained(model, adapter_path)
    if merge:
        model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    _ensure_pad_token(tokenizer, model)
    return model, tokenizer


def load_qlora_model(
    base_model_id: str,
    adapter_path: str,
    device: str | None = None,
) -> Tuple[Any, Any]:
    """Load QLoRA adapter on quantized base model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hw = detect_hardware()
    device = device or hw.device
    bnb_config = _default_bnb_config()

    logger.info("Loading QLoRA: base=%s adapter=%s", base_model_id, adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=get_torch_dtype(hw.dtype),
        trust_remote_code=True,
    ).to(device)
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    _ensure_pad_token(tokenizer, model)
    return model, tokenizer
