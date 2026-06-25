"""
Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models
https://arxiv.org/pdf/2508.00819
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel

if TYPE_CHECKING:
    from dllm.DLLM import DLLMConfig


def _calculate_eos_confidence(
    logits: Tensor,
    total_lengths: Tensor,
    prompt_length: int,
    eos_check_tokens: int,
    eos_token_id: int,
) -> torch.Tensor:
    if eos_token_id is None:
        return torch.zeros(logits.shape[0], device=logits.device)

    confidences = F.softmax(logits, dim=-1)
    predicted_tokens = torch.argmax(logits, dim=-1)
    batch_eos_confidences = []
    for i in range(logits.shape[0]):
        eos_confs_for_avg = []
        start_scan_pos = total_lengths[i].item() - 1
        end_scan_pos = prompt_length - 1
        for pos in range(start_scan_pos, end_scan_pos, -1):
            if len(eos_confs_for_avg) >= eos_check_tokens:
                break
            if predicted_tokens[i, pos] == eos_token_id:
                eos_confs_for_avg.append(confidences[i, pos, eos_token_id].item())
        avg_conf = sum(eos_confs_for_avg) / eos_check_tokens
        batch_eos_confidences.append(avg_conf)
    return torch.tensor(batch_eos_confidences, device=logits.device)


class DAEDALLengthStrategy:
    name = "daedal"

    def __init__(
        self,
        eos_confidence_threshold: float = 0.5,
        expansion_factor: int = 8,
        eos_check_tokens: int = 32,
    ) -> None:
        self.eos_confidence_threshold = eos_confidence_threshold
        self.expansion_factor = expansion_factor
        self.eos_check_tokens = eos_check_tokens
        print(
            "[DAEDALLengthStrategy] initialized with: "
            f"eos_confidence_threshold={eos_confidence_threshold}, "
            f"expansion_factor={expansion_factor}, "
            f"eos_check_tokens={eos_check_tokens}."
        )

    @torch.no_grad()
    def __call__(
        self,
        model: PreTrainedModel,
        prompts: Tensor,
        config: DLLMConfig,
        requested_gen_length: int,
        raw_queries: list[str] | None = None,
    ) -> Tensor:
        with torch.autocast(device_type="cuda"):
            batch_size = prompts.shape[0]
            prompt_length = prompts.shape[1]
            device = prompts.device
            max_gen_length = config.max_gen_length
            is_main_process = (
                not (dist.is_available() and dist.is_initialized())
                or dist.get_rank() == 0
            )

            assert config.eos_id is not None
            if is_main_process:
                print(f"[DAEDAL] predicting lengths for batch_size={batch_size}...")
            gen_lengths = torch.full(
                (batch_size,),
                requested_gen_length,
                dtype=torch.long,
                device=device,
            )
            x = torch.full(
                (batch_size, prompt_length + requested_gen_length),
                config.mask_id,
                dtype=torch.long,
                device=device,
            )
            x[:, :prompt_length] = prompts.clone()

            while True:
                total_lengths = prompt_length + gen_lengths
                max_len_pre = x.shape[1]
                arange_tensor_pre = torch.arange(max_len_pre, device=device).expand(batch_size, -1)
                attention_mask_pre = arange_tensor_pre < total_lengths.unsqueeze(1)
                logits_pre = model(x, attention_mask=attention_mask_pre).logits
                batch_eos_confidences = _calculate_eos_confidence(
                    logits_pre,
                    total_lengths,
                    prompt_length,
                    self.eos_check_tokens,
                    config.eos_id,
                )
                del logits_pre

                sequences_to_expand = (
                    (batch_eos_confidences < self.eos_confidence_threshold)
                    & (gen_lengths < max_gen_length)
                )
                if not sequences_to_expand.any():
                    if is_main_process:
                        print(
                            f"All sequences' EOS confidence reach the threshold "
                            f"{self.eos_confidence_threshold} or max length."
                        )
                    break

                if is_main_process:
                    print(
                        "Some sequences' EOS confidence "
                        f"({[round(c.item(), 4) for c in batch_eos_confidences]}) "
                        f"< {self.eos_confidence_threshold}. Expand initial length."
                    )

                new_gen_lengths = gen_lengths.clone()
                new_gen_lengths[sequences_to_expand] = torch.clamp(
                    gen_lengths[sequences_to_expand] + self.expansion_factor,
                    max=max_gen_length,
                )
                if new_gen_lengths.max() <= gen_lengths.max():
                    if is_main_process:
                        print(
                            "WARNING: Cannot expand initial length further "
                            f"(already at max length: {max_gen_length})."
                        )
                    break

                max_new_total_len = prompt_length + new_gen_lengths.max()
                new_x_tensor = torch.full(
                    (batch_size, max_new_total_len),
                    config.eos_id,
                    dtype=torch.long,
                    device=device,
                )
                for i in range(batch_size):
                    original_total_len = prompt_length + gen_lengths[i].item()
                    new_x_tensor[i, :original_total_len] = x[i, :original_total_len]
                    if sequences_to_expand[i]:
                        new_total_len_i = prompt_length + new_gen_lengths[i].item()
                        new_x_tensor[i, original_total_len:new_total_len_i] = config.mask_id
                x = new_x_tensor
                gen_lengths = new_gen_lengths

            adjusted_gen_lengths = gen_lengths + int(self.eos_check_tokens / 2)
            adjusted_gen_lengths = torch.clamp(
                adjusted_gen_lengths,
                max=config.max_gen_length,
            )
            print(f"[DAEDAL] determines lengths {adjusted_gen_lengths.tolist()}...")
            return adjusted_gen_lengths
