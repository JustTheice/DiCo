import os
from typing import Literal
from pathlib import Path

import torch
import numpy as np

from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset

# from visualizer import get_local

from dllm.utils import set_seed
from dllm.tactics import DAEDALLengthStrategy
from dllm.tactics.cache_backend import CacheBackend, NoCacheBackend, FastDLLMCacheBackend, DKVCacheBackend, build_cache_backend
from dllm.tactics.length_strategy.base import LengthStrategy
from dllm.DLLM import DLLM, DLLMConfig, GenerateOutput
from dllm.recorder.recorder import MetricRecorder, StateTraceRecorder
from dataclasses import dataclass


@dataclass
class BaselineConfig(DLLMConfig):
    remasking: Literal["random", "low_confidence"] = "low_confidence"
    decoding_method: Literal["topk", "factor", "fixed", "entropy_bound"] = "topk"
    k:int = 1
    factor:float = 1.0
    confidence_threshold:float = 0.9
    entropy_bound_gamma: float = 0.1


class DLLMBaseline(DLLM):
    """
        DLLMBaseline
        especially focusing on 'low-confidence' self.remasking
    """

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            config: DLLMConfig,
    ) -> None:
        super().__init__(model, tokenizer, config)

    @classmethod
    def build(
        cls,
        model_path: str,
        config: BaselineConfig | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        mask_id: int | None = None,
        token_overrides: dict | None = None,
        length_strategy: str | LengthStrategy = "vanilla",
        length_strategy_kwargs: dict | None = None,
        cache_backend: str | CacheBackend = "none",
    ):
        if config is None:
            config = BaselineConfig()
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
            records=['metrics'],
            output_attentions=False,
            output_hidden_states=False,
            **kwargs
    ) -> GenerateOutput:

        config = self.config
        assert gen_length <= config.max_gen_length, f"gen_length must <= max_gen_length({config.max_gen_length})"
        assert max_steps <= config.max_steps, f"max_steps must <= max_steps({config.max_steps})"

        print(
            f"decoding method: {config.decoding_method}, k={config.k}, factor={config.factor}, "
            f"confidence_threshold={config.confidence_threshold}, entropy_bound_gamma={config.entropy_bound_gamma}."
        )

        batch = prompt.shape[0]
        prompt_len = prompt.shape[1]
        assert batch == 1, "currently only support batch_size = 1"

        metric_recorder = MetricRecorder()
        state_trace_recorder = StateTraceRecorder()
        if 'metrics' in records:
            metric_recorder.on_generate_start()
        if 'state_trace' in records:
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
            block_mask = self.build_block_mask(x, curr_decoding_pos, total_lengths, block_length)
            step = cache_session.forward_step(
                x,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            x0 = step.x0
            confidences = step.token_confidence

            # 解码策略
            effective_mask = block_mask & mask_token_mask
            transfer_mask, used_fallback = self._build_transfer_mask(
                confidences,
                effective_mask,
                step.logits,
            )
            if used_fallback:
                fallback_steps.append(metric_recorder.accumulated_steps)

            # 更新信息
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

            # update recorder
            if 'metrics' in records:
                # print(f"step {metric_recorder.accumulated_steps} over")
                metric_recorder.on_step_end()
            if 'state_trace' in records:
                state_trace_recorder.on_step_end(
                    x0,
                    confidences,
                    transfer_mask,
                    step.hidden_states,
                    step.attentions,
                )

        # compute recorder
        if 'metrics' in records:
            metric_recorder.on_generate_end(gen_length=gen_length, max_steps=gen_length)
        if 'state_trace' in records:
            state_trace_recorder.on_generate_end()
            state_trace_recorder.record['fallback_steps'] = fallback_steps

        # 把steps_hidden_states存到本地文件
        # np.save(f'exp/output/hidden_states_baseline.npy', np.array(steps_hidden_states))

        return GenerateOutput(
            out=x,
            state_trace=state_trace_recorder.record,
            metrics=metric_recorder.record,
        )


def main():
    # set_seed(1234)
    device = 'cuda:5'

    # gsm8k prompt
    gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
    questions = gsm8k_dataset['test']['question'][0:3]

    # use llada
    # model_path = "/home/anyilin/works/dllm-research/models/LLaDA-8B-Instruct"
    # # model_path = "/home/xiangzhong_ayl/dllm/works/dllm-research/models/LLaDA-8B-Instruct"
    # # model_path = "/root/autodl-tmp/dllm-research/models/LLaDA-8B-Instruct"
    # mask_id = 126336

    # dream
    model_path = "/home/anyilin/works/dllm-research/models/Dream-7B-Instruct"
    mask_id=151666
    # 如果需要改默认超参数，可显式传入 BaselineConfig(...)

    prompt_prefix = ""

    gen_length = 128
    block_length = 128
    sampler = DLLMBaseline.build(
        model_path=model_path,
        device=device,
        torch_dtype=torch.bfloat16,
        mask_id=mask_id,
        config=BaselineConfig(
            decoding_method='fixed', confidence_threshold=0.9, k=1, entropy_bound_gamma=0.1
        ),
        # cache_backend=DKVCacheBackend(mode="prefix-decode", cache_reloading_step=32),
        # cache_backend=FastDLLMCacheBackend(mode="prefix")
    )
    tokenizer = sampler.tokenizer
    
    for i, raw_query in enumerate(questions):
        prompt_text = prompt_prefix + raw_query
        print('=' * 20 + f" Generating prompt_idx: {i} " + '=' * 20)
        print(f"Prompt_{i}: {prompt_text}\n")

        m = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

        # state_trace
        OUT = sampler.generate(
            prompt=input_ids, 
            gen_length=gen_length, 
            max_steps=gen_length, 
            block_length=block_length,
            raw_queries=[raw_query],
            records=['metrics']
        )
        out = OUT.out
        ans = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        print(f"Prompt_{i}'s answer: {ans}\n")
        print(f"Generation Metrics: {OUT.metrics}\n")
        # print(f"hidden_states shape: {OUT.state_trace['hidden_states_all'].shape}\n")
        # print(f"attentions shape: {OUT.state_trace['attentions_all'].shape}\n")

    #     # 将hidden states保存到本地文件
    #     # np.save(f'exp/huashan2/rawdata/hidden_states_gsm8k_pmt{i}.npy', OUT.state_trace['hidden_states_all'])


if __name__ == '__main__':
    main()
