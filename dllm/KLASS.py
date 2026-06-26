"""
KLASS: KL-Guided Fast Inference in Masked Diffusion Models
https://arxiv.org/abs/2511.05664
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from dllm.DLLM import GenerateOutput
from dllm.DLLMBaseline import DLLMBaseline, BaselineConfig
from dllm.recorder.recorder import MetricRecorder, StateTraceRecorder
from dllm.tactics.cache_backend import CacheBackend
from dllm.tactics.length_strategy.base import LengthStrategy
from dllm.utils import set_seed


@dataclass
class KLASSConfig(BaselineConfig):
    kl_threshold: float = 0.01
    kl_history_length: int = 2


class KLASS(DLLMBaseline):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: KLASSConfig,
    ) -> None:
        super().__init__(model, tokenizer, config)

    @classmethod
    def build(
        cls,
        model_path: str,
        config: KLASSConfig | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        mask_id: int | None = None,
        token_overrides: dict | None = None,
        length_strategy: str | LengthStrategy = "vanilla",
        length_strategy_kwargs: dict | None = None,
        cache_backend: str | CacheBackend = "none",
    ):
        if config is None:
            config = KLASSConfig()
        return super().build(
            model_path=model_path,
            config=config,
            device=device,
            torch_dtype=torch_dtype,
            mask_id=mask_id,
            token_overrides=token_overrides,
            length_strategy=length_strategy,
            length_strategy_kwargs=length_strategy_kwargs,
            cache_backend=cache_backend,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt,
        gen_length=256,
        max_steps=256,
        block_length=256,
        raw_queries=None,
        records=["metrics"],
        output_attentions=False,
        output_hidden_states=False,
        **kwargs,
    ) -> GenerateOutput:
        config = self.config
        assert gen_length <= config.max_gen_length, f"gen_length must <= max_gen_length({config.max_gen_length})"
        assert max_steps <= config.max_steps, f"max_steps must <= max_steps({config.max_steps})"

        print(
            f"decoding method: {config.decoding_method}, k={config.k}, factor={config.factor}, "
            f"confidence_threshold={config.confidence_threshold}, entropy_bound_gamma={config.entropy_bound_gamma}, "
            f"kl_threshold={config.kl_threshold}, kl_history_length={config.kl_history_length}."
        )

        batch = prompt.shape[0]
        prompt_len = prompt.shape[1]
        assert batch == 1, "currently only support batch_size = 1"

        metric_recorder = MetricRecorder()
        state_trace_recorder = StateTraceRecorder()
        if "metrics" in records:
            metric_recorder.on_generate_start()
        if "state_trace" in records:
            state_trace_recorder.on_generate_start(prompt_len=prompt_len)

        state = self.prepare_generation_state(prompt, gen_length, block_length, raw_queries=raw_queries)
        gen_length = state.gen_length
        x = state.x
        attention_mask = state.attention_mask
        prompt_mask = state.prompt_mask
        curr_decoding_pos = state.curr_decoding_pos
        mask_token_mask = state.mask_token_mask
        total_lengths = state.total_lengths
        cache_session = self.cache_backend.start_session(self)
        active_block = None

        p_prev_block = None
        kl_history_block = None
        block_klass_step = 0
        fallback_steps = []
        while mask_token_mask.any():
            block_start = int(curr_decoding_pos[0].item())
            block_end = min(block_start + block_length, total_lengths[0].item())
            if active_block != (block_start, block_end):
                if active_block is not None:
                    cache_session.finish_block()
                cache_session.start_block(
                    x,
                    block_start,
                    block_end,
                    total_lengths,
                    prompt_mask,
                    attention_mask=attention_mask,
                )
                active_block = (block_start, block_end)
                p_prev_block = None
                kl_history_block = None
                block_klass_step = 0

            block_mask = self.build_block_mask(x, curr_decoding_pos, total_lengths, block_length)
            effective_mask = block_mask & mask_token_mask
            step = cache_session.forward_step(
                x,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            x0 = step.x0
            confidences = step.token_confidence

            # (B, block_length, vocab_size)
            p_curr_block = F.softmax(step.logits[:, block_start:block_end, :].to(torch.float64), dim=-1)
            if p_prev_block is None:
                p_prev_block = torch.zeros_like(p_curr_block)
                kl_history_block = torch.zeros(
                    (*p_curr_block.shape[:2], config.kl_history_length),
                    dtype=torch.float64,
                    device=x.device,
                )   # (B, block_length, kl_history_length)
            kl_current_prev = (
                p_curr_block
                * (torch.log(p_curr_block + 1e-12) - torch.log(p_prev_block + 1e-12))
            ).sum(dim=-1)
            kl_history_block = torch.roll(kl_history_block, shifts=-1, dims=-1)
            kl_history_block[..., -1] = kl_current_prev
            p_prev_block = p_curr_block.clone()

            if block_klass_step >= config.kl_history_length - 1:
                stable_mask_block = torch.all(kl_history_block < config.kl_threshold, dim=-1)
            else:
                stable_mask_block = torch.zeros_like(kl_current_prev, dtype=torch.bool)
            stable_mask = torch.zeros_like(effective_mask)
            stable_mask[:, block_start:block_end] = stable_mask_block
            reliable_mask = effective_mask & stable_mask

            transfer_mask, used_fallback = self._build_transfer_mask(
                confidences,
                reliable_mask,
                step.logits,
                enable_fallback=False,
            )

            if not transfer_mask.any():
                effective_confidences = confidences.masked_fill(~effective_mask, -torch.inf)
                transfer_mask = torch.zeros_like(effective_mask, dtype=torch.bool, device=effective_mask.device)
                for b in range(batch):
                    _, select_index = torch.topk(effective_confidences[b], k=1)
                    transfer_mask[b, select_index] = True
                used_fallback = True
            if used_fallback:
                fallback_steps.append(metric_recorder.accumulated_steps)

            x[transfer_mask] = x0[transfer_mask]
            cache_session.on_tokens_updated(x, transfer_mask)
            mask_token_mask = (x == config.mask_id)
            curr_decoding_pos = self.advance_decoding_position(
                x, curr_decoding_pos, total_lengths, block_length, config.mask_id
            )
            new_block_start = int(curr_decoding_pos[0].item())
            if active_block is not None and (not mask_token_mask.any() or new_block_start != active_block[0]):
                cache_session.finish_block()
                active_block = None
            block_klass_step += 1

            if "metrics" in records:
                metric_recorder.on_step_end()
            if "state_trace" in records:
                state_trace_recorder.on_step_end(
                    x0,
                    confidences,
                    transfer_mask,
                    step.hidden_states,
                    step.attentions,
                )

        if "metrics" in records:
            metric_recorder.on_generate_end(gen_length=gen_length, max_steps=gen_length)
        if "state_trace" in records:
            state_trace_recorder.on_generate_end()
            state_trace_recorder.record["fallback_steps"] = fallback_steps

        return GenerateOutput(
            out=x,
            state_trace=state_trace_recorder.record,
            metrics=metric_recorder.record,
        )

