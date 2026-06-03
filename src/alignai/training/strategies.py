"""Fine-tuning strategy configurations for FFT, LoRA, and QLoRA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from alignai.config import get_config
from alignai.hardware import detect_hardware, get_torch_dtype


@dataclass
class TrainingHyperparams:
    """Hyperparameter configuration for fine-tuning jobs."""

    strategy: str = "lora"  # full | lora | qlora
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    gradient_accumulation_steps: int = 4
    save_steps: int = 100
    logging_steps: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_seq_length": self.max_seq_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }


def get_base_model_id(strategy: str) -> str:
    """Return appropriate base model for strategy."""
    cfg = get_config()
    if strategy == "full":
        return cfg.base_model_fft
    return cfg.base_model_lora


def build_lora_config(hparams: TrainingHyperparams) -> Any:
    """Build PEFT LoRA configuration."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=hparams.lora_r,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        target_modules=hparams.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def build_bnb_config() -> Any:
    """Build bitsandbytes 4-bit quantization config for QLoRA."""
    from transformers import BitsAndBytesConfig

    hw = detect_hardware()
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=get_torch_dtype(hw.dtype),
        bnb_4bit_use_double_quant=True,
    )


def prepare_model_for_training(
    model_id: str,
    strategy: str,
    hparams: TrainingHyperparams,
) -> tuple[Any, Any, Optional[Any]]:
    """Load and prepare model, tokenizer, and optional PEFT config."""
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hw = detect_hardware()
    dtype = get_torch_dtype(hw.dtype)
    peft_config = None

    if strategy == "qlora":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=build_bnb_config(),
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        peft_config = build_lora_config(hparams)
        model = get_peft_model(model, peft_config)
    elif strategy == "lora":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
        )
        peft_config = build_lora_config(hparams)
        model = get_peft_model(model, peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, peft_config
