"""
https://arxiv.org/abs/2505.22618
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from .base import CacheBackend, CacheSession

if TYPE_CHECKING:
    from dllm.DLLM import DLLM, ForwardStepOutput


class FastDLLMCacheSession(CacheSession):
    def __init__(self, sampler: DLLM, mode: Literal["prefix", "dual"]):
        super().__init__(sampler)
        self.mode = mode
        self.past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None
        self.past_cache_positions_mask: torch.Tensor | None = None
        self.full_logits: torch.Tensor | None = None
        self.is_warmed = False

    def start_block(
        self,
        x: torch.Tensor,
        block_start: int,
        block_end: int,
        total_lengths: torch.Tensor,
        prompt_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        self.block_start = int(block_start)
        self.block_end = int(block_end)
        self.total_lengths = total_lengths
        self.prompt_mask = prompt_mask
        self.attention_mask = attention_mask
        self.past_key_values = None
        self.full_logits = None
        self.is_warmed = False
        self.past_cache_positions_mask = torch.zeros_like(x, dtype=torch.bool)
        self.past_cache_positions_mask[:, :self.block_start] = True
        if self.mode == "dual":
            self.past_cache_positions_mask[:, self.block_end:] = True
        if self.sampler.cfg_scale > 0.0:
            raise NotImplementedError("FastDLLMCacheBackend does not support cfg_scale > 0 yet.")

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

        if not self.is_warmed:
            return self._warm_full_step(
                x,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_probs=return_probs,
            )

        if self.mode == "prefix":
            return self._forward_prefix_refine(
                x,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_probs=return_probs,
            )
        return self._forward_dual_refine(
            x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_probs=return_probs,
        )

    def on_tokens_updated(self, x: torch.Tensor, transfer_mask: torch.Tensor) -> None:
        pass

    def finish_block(self) -> None:
        self.block_start = None
        self.block_end = None
        self.total_lengths = None
        self.prompt_mask = None
        self.attention_mask = None
        self.past_key_values = None
        self.past_cache_positions_mask = None
        self.full_logits = None
        self.is_warmed = False

    def _warm_full_step(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_probs: bool = False,
    ) -> ForwardStepOutput:
        output, logits, hidden_states, attentions = self.sampler.model_forward(
            x,
            self.prompt_mask,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        self.past_key_values = self._compress_past_key_values(output.past_key_values, self.past_cache_positions_mask)
        self.full_logits = logits
        self.is_warmed = True
        return self.build_full_step_output(
            x,
            x,
            logits,
            torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1),
            hidden_states=hidden_states,
            attentions=attentions,
            return_probs=return_probs,
            logits_base=logits,
        )

    def _forward_prefix_refine(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_probs: bool = False,
    ) -> ForwardStepOutput:
        x_local = x[:, self.block_start:]
        _, logits, hidden_states, attentions = self.sampler.model_forward(
            x_local,
            self.prompt_mask,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            past_cache_positions_mask=self.past_cache_positions_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        positions = torch.arange(self.block_start, x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        return self.build_full_step_output(
            x,
            x_local,
            logits,
            positions,
            hidden_states=hidden_states,
            attentions=attentions,
            return_probs=return_probs,
        )

    def _forward_dual_refine(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_probs: bool = False,
    ) -> ForwardStepOutput:
        x_local = x[:, self.block_start:self.block_end]
        _, logits, hidden_states, attentions = self.sampler.model_forward(
            x_local,
            self.prompt_mask,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            past_cache_positions_mask=self.past_cache_positions_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        self.full_logits[:, self.block_start:self.block_end] = logits
        positions = torch.arange(self.block_start, self.block_end, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        return self.build_full_step_output(
            x,
            x_local,
            logits,
            positions,
            hidden_states=hidden_states,
            attentions=attentions,
            return_probs=return_probs,
            logits_base=self.full_logits,
        )

    def _compress_past_key_values(
        self,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        positions: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        cache_positions = torch.nonzero(positions, as_tuple=False)[:, 1].view(positions.shape[0], -1)
        compressed = []
        for past_key, past_value in past_key_values:
            gather_index = cache_positions[:, None, :, None].expand(
                past_key.shape[0],
                past_key.shape[1],
                cache_positions.shape[1],
                past_key.shape[-1],
            )
            compressed.append(
                (
                    past_key.gather(2, gather_index),
                    past_value.gather(2, gather_index),
                )
            )
        return tuple(compressed)


class FastDLLMCacheBackend(CacheBackend):
    def __init__(self, mode: Literal["prefix", "dual"] = "prefix"):
        self.mode = mode
        print("[FastDLLMCacheBackend] initialized with: " f"mode={self.mode}.")

    def start_session(self, sampler: DLLM) -> FastDLLMCacheSession:
        return FastDLLMCacheSession(sampler, mode=self.mode)
