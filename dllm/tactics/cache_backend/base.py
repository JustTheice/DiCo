from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from dllm.utils import add_gumbel_noise

if TYPE_CHECKING:
    from dllm.DLLM import DLLM, ForwardStepOutput


class CacheBackend(ABC):
    @abstractmethod
    def start_session(self, sampler: DLLM) -> CacheSession:
        raise NotImplementedError


class CacheSession(ABC):
    def __init__(self, sampler: DLLM):
        self.sampler = sampler
        self.prompt_mask: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None
        self.block_start: int | None = None
        self.block_end: int | None = None
        self.total_lengths: torch.Tensor | None = None

    @abstractmethod
    def start_block(
        self,
        x: torch.Tensor,
        block_start: int,
        block_end: int,
        total_lengths: torch.Tensor,
        prompt_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward_step(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_probs: bool = False,
    ) -> ForwardStepOutput:
        raise NotImplementedError

    @abstractmethod
    def on_tokens_updated(self, x: torch.Tensor, transfer_mask: torch.Tensor) -> None:
        raise NotImplementedError

    def build_full_step_output(
        self,
        x_full: torch.Tensor,
        x_local: torch.Tensor,
        logits_local: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: tuple[torch.Tensor, ...] | None = None,
        attentions: tuple[torch.Tensor, ...] | None = None,
        return_probs: bool = False,
        logits_base: torch.Tensor | None = None,
    ) -> ForwardStepOutput:
        logits_with_noise = add_gumbel_noise(logits_local, self.sampler.temperature)
        x0_local = torch.argmax(logits_with_noise, dim=-1)
        probs_local = F.softmax(logits_local, dim=-1)
        x0_p_local = torch.gather(probs_local, dim=-1, index=x0_local.unsqueeze(-1)).squeeze(-1)
        confidence_local = torch.where(x_local == self.sampler.mask_id, x0_p_local, -np.inf)

        x0 = x_full.clone()
        x0.scatter_(1, positions, x0_local)
        token_confidence = torch.full(
            x_full.shape,
            -np.inf,
            dtype=x0_p_local.dtype,
            device=x0_p_local.device,
        )
        token_confidence.scatter_(1, positions, x0_p_local)
        confidence = torch.full_like(token_confidence, -np.inf)
        confidence.scatter_(1, positions, confidence_local)

        if logits_base is None:
            logits = torch.zeros(
                (x_full.shape[0], x_full.shape[1], logits_local.shape[-1]),
                dtype=logits_local.dtype,
                device=logits_local.device,
            )
        else:
            logits = logits_base.clone()
        logits.scatter_(
            1,
            positions[:, :, None].expand(x_full.shape[0], positions.shape[1], logits_local.shape[-1]),
            logits_local,
        )

        probs = None
        if return_probs:
            probs = torch.zeros_like(logits)
            probs.scatter_(
                1,
                positions[:, :, None].expand(x_full.shape[0], positions.shape[1], logits_local.shape[-1]),
                probs_local,
            )

        from dllm.DLLM import ForwardStepOutput

        return ForwardStepOutput(
            x0=x0,
            confidence=confidence,
            token_confidence=token_confidence,
            logits=logits,
            probs=probs,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def restart_block(
        self,
        x: torch.Tensor,
        block_start: int,
        block_end: int,
        total_lengths: torch.Tensor,
        prompt_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        self.finish_block()
        self.start_block(
            x,
            block_start,
            block_end,
            total_lengths,
            prompt_mask,
            attention_mask=attention_mask,
        )

    @abstractmethod
    def finish_block(self) -> None:
        raise NotImplementedError
