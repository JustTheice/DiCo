'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
from typing import Any

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.DLLM import GenerateOutput
from dllm.DiCo import DiCo, DiCoConfig
from eval.eval_model.eval_base import set_seed, BaseEvalHarness
from eval.eval_model.utils import build_config_from_kwargs


@register_model("eval_sampler")
class MRSamplerEvalHarness(BaseEvalHarness):
    def __init__(
            self,
            model_path: str = './model_cache',
            device="cuda",
            batch_size=1,
            mc_num=128,
            steps=256,
            gen_length=256,
            block_length=256,
            **kwargs,
    ):
        record_timing_breakdown = kwargs.pop("record_timing_breakdown", 0)
        if isinstance(record_timing_breakdown, str):
            record_timing_breakdown = bool(int(record_timing_breakdown))
        records = ["timing_breakdown"] if record_timing_breakdown else []
        self.overall_timing_breakdowns = []
        super().__init__(
            model_path,
            batch_size,
            mc_num,
            steps,
            gen_length,
            block_length,
            sampler_cls=DiCo,
            sampler_config=build_config_from_kwargs(DiCoConfig, kwargs),
            records=records,
            **kwargs,
        )

    def _on_generate_output(self, out: GenerateOutput) -> None:
        if "timing_breakdown" in self.records:
            self.overall_timing_breakdowns.append(out.state_trace["timing_breakdown"])

    def _gather_extra_records(self) -> None:
        if "timing_breakdown" in self.records and self.accelerator is not None:
            self.overall_timing_breakdowns = self.accelerator.gather_for_metrics(self.overall_timing_breakdowns)

    def _build_extra_report(self) -> dict[str, Any]:
        if "timing_breakdown" not in self.records or not self.overall_timing_breakdowns:
            return {}
        return {
            "timing_breakdown": {
                "summary": {
                    "total_time": sum(item["summary"]["total_time"] for item in self.overall_timing_breakdowns),
                    "model_forward_time": sum(item["summary"]["model_forward_time"] for item in self.overall_timing_breakdowns),
                    "other_time": sum(item["summary"]["other_time"] for item in self.overall_timing_breakdowns),
                    "dico_time": sum(item["summary"]["dico_time"] for item in self.overall_timing_breakdowns),
                },
                "Divide": {
                    "total_time": sum(item["Divide"]["total_time"] for item in self.overall_timing_breakdowns),
                    "model_forward_time": sum(item["Divide"]["model_forward_time"] for item in self.overall_timing_breakdowns),
                    "seed_identification_time": sum(item["Divide"]["seed_identification_time"] for item in self.overall_timing_breakdowns),
                    "clusters_formation_time": sum(item["Divide"]["clusters_formation_time"] for item in self.overall_timing_breakdowns),
                    "other_time": sum(item["Divide"]["other_time"] for item in self.overall_timing_breakdowns),
                },
                "Conquer": {
                    "total_time": sum(item["Conquer"]["total_time"] for item in self.overall_timing_breakdowns),
                    "model_forward_time": sum(item["Conquer"]["model_forward_time"] for item in self.overall_timing_breakdowns),
                    "boundary_adjustment_time": sum(item["Conquer"]["boundary_adjustment_time"] for item in self.overall_timing_breakdowns),
                    "other_time": sum(item["Conquer"]["other_time"] for item in self.overall_timing_breakdowns),
                },
                "Finalize": {
                    "total_time": sum(item["Finalize"]["total_time"] for item in self.overall_timing_breakdowns),
                    "model_forward_time": sum(item["Finalize"]["model_forward_time"] for item in self.overall_timing_breakdowns),
                    "other_time": sum(item["Finalize"]["other_time"] for item in self.overall_timing_breakdowns),
                },
            }
        }


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
