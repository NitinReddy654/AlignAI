"""Text generation with latency and token metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List

from alignai.logging_utils import setup_logging

logger = setup_logging(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are AlignAI Support Assistant, a professional enterprise support agent. "
    "Provide accurate, concise, and helpful responses. Follow company policies, "
    "maintain a professional tone, and prioritize user safety."
)


@dataclass
class GenerationResult:
    """Single generation output with performance metrics."""

    response: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "latency_ms": round(self.latency_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": round(
                self.output_tokens / (self.latency_ms / 1000.0), 2
            )
            if self.latency_ms > 0
            else 0,
        }


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: List[dict],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> GenerationResult:
    """Generate a conversational response with metrics."""
    import torch

    model.eval()
    device = next(model.parameters()).device

    formatted = []
    if messages and messages[0].get("role") != "system":
        formatted.append({"role": "system", "content": system_prompt})
    formatted.extend(messages)

    input_ids = tokenizer.apply_chat_template(
        formatted,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    input_token_count = input_ids.shape[-1]
    attention_mask = torch.ones_like(input_ids)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    response_ids = outputs[0][input_ids.shape[-1] :]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    output_token_count = len(response_ids)

    return GenerationResult(
        response=response,
        latency_ms=latency_ms,
        input_tokens=input_token_count,
        output_tokens=output_token_count,
        total_tokens=input_token_count + output_token_count,
    )
