from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch
from transformers import PreTrainedModel

if TYPE_CHECKING:
    from dllm.DLLM import DLLMConfig


class LengthStrategy(Protocol):
    name: str

    def __call__(
        self,
        model: PreTrainedModel,
        prompts: torch.Tensor,
        config: DLLMConfig,
        requested_gen_length: int,
        raw_queries: list[str] | None = None,
    ) -> torch.Tensor:
        ...
