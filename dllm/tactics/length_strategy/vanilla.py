from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import PreTrainedModel

if TYPE_CHECKING:
    from dllm.DLLM import DLLMConfig


class VanillaLengthStrategy:
    name = "vanilla"

    def __init__(self) -> None:
        print("[VanillaLengthStrategy] initialized with: no parameters.")

    def __call__(
        self,
        model: PreTrainedModel,
        prompts: torch.Tensor,
        config: DLLMConfig,
        requested_gen_length: int,
        raw_queries: list[str] | None = None,
    ) -> torch.Tensor:
        batch_size = prompts.shape[0]
        return torch.full(
            (batch_size,),
            requested_gen_length,
            dtype=torch.long,
            device=prompts.device,
        )
