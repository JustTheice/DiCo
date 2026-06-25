"""
https://proceedings.neurips.cc/paper_files/paper/2025/hash/db0ee27cb50dd9087b133f6e7d28a90e-Abstract-Conference.html
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from .base import CacheBackend, CacheSession

if TYPE_CHECKING:
    from dllm.DLLM import DLLM, ForwardStepOutput


class DKVCacheSession(CacheSession):
    def __init__(self, sampler: DLLM, mode: Literal["decode", "prefix-decode"], cache_reloading_step: int):
        super().__init__(sampler)
        self.mode = mode
        self.cache_reloading_step = cache_reloading_step
        self.past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None
        self.past_cache_positions_mask: torch.Tensor | None = None
        self.target_cache_positions_mask: torch.Tensor | None = None
        self.step_idx = 0

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
        self.past_cache_positions_mask = None
        self.target_cache_positions_mask = None
        self.step_idx = 0
        if self.sampler.cfg_scale > 0.0:
            raise NotImplementedError("DKVCacheBackend does not support cfg_scale > 0 yet.")

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

        if self.step_idx == 0 or self._is_reload_step():
            return self._warm_full_step(
                x,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_probs=return_probs,
            )
        return self._forward_active_refine(
            x,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_probs=return_probs,
        )

    def on_tokens_updated(self, x: torch.Tensor, transfer_mask: torch.Tensor) -> None:
        target_cache_positions_mask = transfer_mask.clone()
        if self.past_cache_positions_mask is not None:
            target_cache_positions_mask |= self.past_cache_positions_mask
        target_cache_positions_mask |= self.prompt_mask
        if self.attention_mask is not None:
            target_cache_positions_mask &= self.attention_mask
        self.target_cache_positions_mask = target_cache_positions_mask
        self.step_idx += 1

    def finish_block(self) -> None:
        self.block_start = None
        self.block_end = None
        self.total_lengths = None
        self.prompt_mask = None
        self.attention_mask = None
        self.past_key_values = None
        self.past_cache_positions_mask = None
        self.target_cache_positions_mask = None
        self.step_idx = 0

    def _is_reload_step(self) -> bool:
        return self.step_idx == 1 or self.step_idx % self.cache_reloading_step == 0

    def _warm_full_step(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_probs: bool = False,
    ) -> ForwardStepOutput:
        x_local = x
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        if self.step_idx == 0:
            self.target_cache_positions_mask = self.prompt_mask
        model_kwargs = {
            "attention_mask": attention_mask,
            "use_cache": True,
            "output_hidden_states": output_hidden_states,
            "output_attentions": output_attentions,
        }
        if self.mode == "prefix-decode" and self.step_idx > 0:
            prompt_len = int(self.prompt_mask[0].sum().item())
            x_local = x[:, prompt_len:]
            positions = torch.arange(prompt_len, x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            model_kwargs["past_key_values"] = tuple(
                (
                    past_key[:, :, :prompt_len],
                    past_value[:, :, :prompt_len],
                )
                for past_key, past_value in self.past_key_values
            )
            model_kwargs["past_cache_positions_mask"] = self.prompt_mask

        output, logits, hidden_states, attentions = self.sampler.model_forward(
            x_local,
            self.prompt_mask,
            **model_kwargs,
        )
        self.past_key_values = self._compress_past_key_values(output.past_key_values, self.target_cache_positions_mask)
        self.past_cache_positions_mask = self.target_cache_positions_mask
        return self.build_full_step_output(
            x,
            x_local,
            logits,
            positions,
            hidden_states=hidden_states,
            attentions=attentions,
            return_probs=return_probs,
            logits_base=logits if x_local is x else None,
        )

    def _forward_active_refine(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_probs: bool = False,
    ) -> ForwardStepOutput:
        current_positions = torch.nonzero(~self.past_cache_positions_mask, as_tuple=False)[:, 1].view(x.shape[0], -1)
        x_local = x[~self.past_cache_positions_mask].view(x.shape[0], -1)
        output, logits, hidden_states, attentions = self.sampler.model_forward(
            x_local,
            self.prompt_mask,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            past_cache_positions_mask=self.past_cache_positions_mask,
            use_cache=True,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        self.past_key_values = self._compress_past_key_values(output.past_key_values, self.target_cache_positions_mask)
        self.past_cache_positions_mask = self.target_cache_positions_mask
        return self.build_full_step_output(
            x,
            x_local,
            logits,
            current_positions,
            hidden_states=hidden_states,
            attentions=attentions,
            return_probs=return_probs,
        )

    def _compress_past_key_values(
        self,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        positions_mask: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        cache_positions = torch.nonzero(positions_mask, as_tuple=False)[:, 1].view(positions_mask.shape[0], -1)
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


class DKVCacheBackend(CacheBackend):
    def __init__(self, mode: Literal["decode", "prefix-decode"] = "decode", cache_reloading_step: int = 8):
        self.mode = mode
        self.cache_reloading_step = cache_reloading_step
        print(
            "[DKVCacheBackend] initialized with: "
            f"mode={self.mode}, cache_reloading_step={self.cache_reloading_step}."
        )

    def start_session(self, sampler: DLLM) -> DKVCacheSession:
        return DKVCacheSession(sampler, mode=self.mode, cache_reloading_step=self.cache_reloading_step)
