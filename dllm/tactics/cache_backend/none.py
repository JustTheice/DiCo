from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from dllm.utils import add_gumbel_noise

from .base import CacheBackend, CacheSession

if TYPE_CHECKING:
    from dllm.DLLM import DLLM, ForwardStepOutput


class NoCacheSession(CacheSession):
    def start_block(
        self,
        x: torch.Tensor,
        block_start: int,
        block_end: int,
        total_lengths: torch.Tensor,
        prompt_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        self.block_start = block_start
        self.block_end = block_end
        self.total_lengths = total_lengths
        self.prompt_mask = prompt_mask
        self.attention_mask = attention_mask

    def forward_step(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_probs: bool = False,
    ) -> ForwardStepOutput:
        if attention_mask is None:
            attention_mask = self.attention_mask
        _, logits, hidden_states, attentions = self.sampler.model_forward(
            x,
            self.prompt_mask,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        return self.build_full_step_output(
            x,
            logits,
            hidden_states=hidden_states,
            attentions=attentions,
            return_probs=return_probs,
        )

    def on_tokens_updated(self, x: torch.Tensor, transfer_mask: torch.Tensor) -> None:
        pass

    def build_full_step_output(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        hidden_states: tuple[torch.Tensor, ...] | None = None,
        attentions: tuple[torch.Tensor, ...] | None = None,
        return_probs: bool = False,
    ) -> ForwardStepOutput:
        logits_with_noise = add_gumbel_noise(logits, self.sampler.temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        probs = F.softmax(logits, dim=-1)
        token_confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        confidence = torch.where(x == self.sampler.mask_id, token_confidence, -np.inf)
        from dllm.DLLM import ForwardStepOutput

        return ForwardStepOutput(
            x0=x0,
            confidence=confidence,
            token_confidence=token_confidence,
            logits=logits,
            probs=probs if return_probs else None,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def finish_block(self) -> None:
        self.block_start = None
        self.block_end = None
        self.total_lengths = None
        self.prompt_mask = None
        self.attention_mask = None


class NoCacheBackend(CacheBackend):
    def __init__(self) -> None:
        print("[NoCacheBackend] initialized with: no parameters.")

    def start_session(self, sampler: DLLM) -> NoCacheSession:
        return NoCacheSession(sampler)
