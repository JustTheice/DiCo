'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.DLLMBaseline import DLLMBaseline, BaselineConfig
from eval.eval_model.eval_base import BaseEvalHarness, set_seed
from eval.eval_model.utils import build_config_from_kwargs


@register_model("eval_sampler")
class BaselineEvalHarness(BaseEvalHarness):
    def __init__(
        self,
        model_path='./model_cache',
        batch_size=1,
        mc_num=128,
        steps=256,
        gen_length=256,
        block_length=256,
        device="cuda",
        **kwargs,
    ):

        super().__init__(
            model_path,
            batch_size,
            mc_num,
            steps,
            gen_length,
            block_length,
            sampler_cls=DLLMBaseline,
            sampler_config=build_config_from_kwargs(BaselineConfig, kwargs),
            **kwargs,
        )


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
