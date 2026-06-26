import time
from dataclasses import dataclass, field
from typing import List, Any, Dict, Tuple
import numpy as np
import torch
from torch import Tensor
from dllm.DLLM import GenerationMetrics


class CallbackTemplate:
    def on_generate_start(self, **kwargs): pass

    def on_step_end(self, **kwargs): pass

    def on_generate_end(self, **kwargs): pass


class MetricRecorder(CallbackTemplate):

    def __init__(self):
        self.start_time = 0
        self.record = None
        self.accumulated_steps = 0

    def on_generate_start(self, **kwargs):
        self.start_time = time.perf_counter()
        self.accumulated_steps = 0

    def on_step_end(self, **kwargs):
        self.accumulated_steps += 1

    def on_generate_end(self, gen_length, max_steps, **kwargs):
        end_time = time.perf_counter()
        duration = end_time - self.start_time

        self.record = GenerationMetrics(
            use_seconds=duration,
            use_steps=self.accumulated_steps,
            n_gen_tokens=gen_length,
            tokens_per_second=(gen_length / duration) if duration > 0 else 0,
            step_reduction_ratio=max_steps / self.accumulated_steps if self.accumulated_steps > 0 else 0
        )
        # print(f"[Callback] Metrics Computed: {self.metrics}")


class StateTraceRecorder(CallbackTemplate):

    def __init__(self):
        self.prompt_len = 0
        self.outputs_all = []
        self.confidences_all = []
        self.transfer_masks_all = []
        self.hidden_states_all = []
        self.attentions_all = []
        self.record = {}

    def on_generate_start(self, prompt_len, **kwargs):
        self.prompt_len = prompt_len

    def on_step_end(self, 
                    x0:Tensor=None, 
                    confidences:Tensor=None, 
                    transfer_mask:Tensor=None, 
                    hidden_states:Tuple[Tensor, ...]=None, 
                    attentions:Tuple[Tensor, ...]=None,
                    **kwargs
                    ):
        if x0 is not None:
            self.outputs_all.append(x0[0].detach().cpu().numpy())
        if confidences is not None:
            self.confidences_all.append(confidences[0].detach().cpu().to(torch.float32).numpy())
        if transfer_mask is not None:
            self.transfer_masks_all.append(transfer_mask[0].detach().cpu().numpy())
        if hidden_states is not None:
            np_h = np.array([h[0].detach().cpu().to(torch.float32).numpy() for h in hidden_states])  # (n_layers, seq_len, hidden_size)
            self.hidden_states_all.append(np_h)
        if attentions is not None:
            np_attn = np.array([a[0].detach().cpu().to(torch.float32).numpy() for a in attentions])  # (n_layers, n_heads, seq_len, seq_len)
            self.attentions_all.append(np_attn)

    def on_generate_end(self, gen_length=0, keep_prompt=False, **kwargs):
        pad_id = 0
        for i in range(len(self.outputs_all)):
            curr_len = self.outputs_all[i].shape[0]
            if curr_len < self.prompt_len + gen_length:
                pad_len = self.prompt_len + gen_length - curr_len
                self.outputs_all[i] = np.pad(self.outputs_all[i], (0, pad_len), constant_values=pad_id)
                self.confidences_all[i] = np.pad(self.confidences_all[i], (0, pad_len), constant_values=0.0)
                self.transfer_masks_all[i] = np.pad(self.transfer_masks_all[i], (0, pad_len), constant_values=False)
                if len(self.hidden_states_all):
                    self.hidden_states_all[i] = np.pad(self.hidden_states_all[i], ((0, 0), (0, pad_len), (0, 0)), constant_values=pad_id)
                # self.attentions_all[i] = np.pad(self.attentions_all[i], ((0, pad_len), (0, 0), (0, 0)), constant_values=pad_id)
            if not keep_prompt:
                self.outputs_all[i] = self.outputs_all[i][self.prompt_len:]
                self.confidences_all[i] = self.confidences_all[i][self.prompt_len:]
                self.transfer_masks_all[i] = self.transfer_masks_all[i][self.prompt_len:]
                if len(self.hidden_states_all):
                    self.hidden_states_all[i] = self.hidden_states_all[i][:, self.prompt_len:, :]
        
        self.record = {
            "prompt_len": self.prompt_len,
            "outputs_all": np.array(self.outputs_all),
            "confidences_all": np.array(self.confidences_all),
            "transfer_masks_all": np.array(self.transfer_masks_all),
            "hidden_states_all": np.array(self.hidden_states_all),
            "attentions_all": np.array(self.attentions_all)
        }
