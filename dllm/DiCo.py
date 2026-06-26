"""
Divide and Conquer: Accelerating Diffusion-Based Large Language Models via Adaptive Parallel Decoding
https://arxiv.org/abs/2602.23792
"""

from typing import List, Literal
from torch import Tensor

import torch
import time
import math
import numpy as np

from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset

from dllm.DLLM import DLLM, DLLMConfig, GenerationMetrics, GenerateOutput
from dllm.tactics.cache_backend import CacheBackend
from dllm.tactics.length_strategy.base import LengthStrategy
from dllm.utils import set_seed
from dataclasses import dataclass


@dataclass
class DiCoConfig(DLLMConfig):
    TG_alpha: float = 0.5
    TG_beta: float = 0.05
    # Divide phase config
    max_exploration_steps: int = 10
    exploration_N: int = 4
    tolerance_M: int = 2
    exploration_threshold: float = 0.3
    exploration_seed_method: Literal["soft_nms", "regular_interval"] = "soft_nms"
    # Conquer phase config
    acceleration_parallel_method: Literal["fixed", "factor"] = "factor"
    acceleration_threshold: float = 0.9
    acceleration_low_threshold: float = 0.6
    acceleration_factor: float = 1
    # Finalize phase config
    R_gate: float = 0.8
    mopup_margin_threshold: float = 3.0
    mopup_speed: int = 1

    def __post_init__(self):
        self.ur_factor = self.TG_alpha
        self.initial_min_weight = self.TG_beta


class DiCo(DLLM):
    """
        DiCo Sampler：
        1. Divide Phase: construct decoding zones with stable and moderate confidence, do "gentle" parallel decoding.
        2. Conquer Phase: do confidence-based parallel decoding on decoding zones.
        3. Finalize Phase: do margin-based parallel+top1 decoding on global context。
    """
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            config: DiCoConfig
    ) -> None:
        super().__init__(model, tokenizer, config)

    @classmethod
    def build(
        cls,
        model_path: str,
        config: DiCoConfig | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        mask_id: int | None = None,
        token_overrides: dict | None = None,
        length_strategy: str | LengthStrategy = "vanilla",
        length_strategy_kwargs: dict | None = None,
        cache_backend: str | CacheBackend = "none",
    ):
        if config is None:
            config = DiCoConfig()
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


    def _merge_intervals(self, intervals: List[tuple[int, int]]) -> List[tuple[int, int]]:
        """
            merge overlapping intervals
            eg: [(10, 20), (15, 25), (40, 50)] -> [(10, 25), (40, 50)]
        """
        if len(intervals) < 2:
            return intervals
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged_intervals = [sorted_intervals[0]]
        for current_start, current_end in sorted_intervals[1:]:
            mem_start, mem_end = merged_intervals[-1]
            if mem_end + 1 >= current_start:
                merged_intervals[-1] = (mem_start, max(mem_end, current_end))
            else:
                merged_intervals.append((current_start, current_end))
        return merged_intervals

    def _select_seed_indices(self, conf_temp: Tensor, select_N: int):
        if self.exploration_seed_method == "regular_interval":
            seed_indices = []
            segment_starts = [
                self.block_start + i * self.block_length // select_N
                for i in range(select_N)
            ]
            segment_ends = [
                self.block_start + (i + 1) * self.block_length // select_N
                for i in range(select_N)
            ]
            for b in range(conf_temp.shape[0]):
                seeds = []
                for start, end in zip(segment_starts, segment_ends):
                    conf_seg = conf_temp[b, start:end]
                    if conf_seg.numel() == 0:
                        continue
                    offset = torch.argmax(conf_seg).item()
                    idx = start + offset
                    if conf_temp[b, idx] > self.exploration_threshold:
                        seeds.append(idx)
                seed_indices.append(seeds)
        elif self.exploration_seed_method == "soft_nms":
            seed_indices = []
            for b in range(conf_temp.shape[0]):
                conf_b = conf_temp[b].clone()
                idxs = torch.arange(conf_temp.shape[1], device=conf_temp.device, dtype=torch.float32)
                seeds = []
                for _ in range(select_N):
                    idx = torch.argmax(conf_b).item()
                    if conf_b[idx] <= 0 or conf_temp[b, idx] < self.exploration_threshold:
                        break
                    seeds.append(idx)
                    sigma = self.block_length / 2 / 1.5
                    conf_b *= 1 - torch.exp(-torch.pow(torch.abs(idxs - idx), 2) / (sigma ** 2))
                seed_indices.append(seeds)
        else:
            raise ValueError(f"Unsupported exploration_seed_method: {self.exploration_seed_method}")
        select_N = len(seed_indices[0])
        if select_N == 0:
            return None, 0
        return torch.tensor(seed_indices, dtype=torch.long, device=conf_temp.device), select_N


    #  core methods  #
    def Divide_phase(
            self,
            x: Tensor,
            prompt_mask: Tensor,
            cache_session,
            exp_steps: int = 0,
            current_step: int = -1,
            record_timing_breakdown: bool = False,
    ):
        """
        Divide Phase: construct decoding zones with stable and moderate confidence, do "gentle" parallel decoding.
            exp_steps: maximum exploratory iterations
        """
        prompt_len = prompt_mask[0].sum().item()
        pre_demasked_index = (x != self.mask_id)
        memory_intervals = []

        steps_used = 0
        outputs = []
        confidences = []
        transfer_masks = []
        history_intervals = []
        if exp_steps <= 0:
            return x, memory_intervals, steps_used, outputs, confidences, transfer_masks, history_intervals, {
                "total_time": 0.0,
                "model_forward_time": 0.0,
                "seed_identification_time": 0.0,
                "clusters_formation_time": 0.0,
                "other_time": 0.0,
            } if record_timing_breakdown else None

        if record_timing_breakdown:
            torch.cuda.synchronize(x.device)
            model_forward_time = 0.0
            seed_identification_time = 0.0
            clusters_formation_time = 0.0
            phase_start = time.perf_counter()

        no_advance_n = 0
        for exp_step in range(exp_steps):
            if record_timing_breakdown:
                torch.cuda.synchronize(x.device)
                forward_start = time.perf_counter()
            step = cache_session.forward_step(x)
            if record_timing_breakdown:
                torch.cuda.synchronize(x.device)
                model_forward_time += time.perf_counter() - forward_start
            x0 = step.x0
            confidence = step.confidence
            current_step += 1
            steps_used += 1
            GG = False
            should_break = False
            record_intervals = memory_intervals

            # dynamic N
            unmasked_ratio = (x[:, self.block_start: self.block_end] != self.mask_id).sum().item() / self.block_length
            # select_N = max(1, round(self.exploration_N * (np.cos(np.pi/2 * unmasked_ratio))))
            select_N = self.exploration_N

            mask_token_mask = x == self.mask_id
            confidence[:, 0: self.block_start] = confidence[:, self.block_end:] = -np.inf  # semi support
            confidence[~mask_token_mask] = -np.inf

            # mutiplying positional weights
            if self.positional_weights_type == 'ratio':
                # MARK
                global_unmasked_ratio = (x[:, prompt_len: ] != self.mask_id).sum().item() / self.gen_length
                mask_idxs_in_block = torch.where(mask_token_mask[:, self.block_start:self.block_end])[1]
                # dynamic length_for_weighting: from first masked token to block end
                first_mask_idx_in_block = mask_idxs_in_block[0].item() if mask_idxs_in_block.numel() > 0 else self.block_length
                length_for_weighting = self.block_end - (self.block_start + first_mask_idx_in_block)
                # print(f"first_mask_idx_in_block: {first_mask_idx_in_block}, abs: {self.block_start+first_mask_idx_in_block}, length_for_weighting: {length_for_weighting}")
                # print(f"{'!'*10} length_for_weighting: {length_for_weighting}, block_start: {self.block_start}, block_end: {self.block_end}, first_mask_idx_in_block: {first_mask_idx_in_block} {'!'*10}")
                if length_for_weighting == 0:
                    GG = True
                else:
                    dynamic_positional_weights = self.compute_dynamic_positional_weights(length_for_weighting, global_unmasked_ratio, device=x0.device)
                    confidence[:, self.block_start+first_mask_idx_in_block: self.block_end] = confidence[:, self.block_start+first_mask_idx_in_block: self.block_end] * dynamic_positional_weights
            elif self.positional_weights_type == 'static':
                confidence[:, self.block_start: self.block_end] = confidence[:, self.block_start: self.block_end] * self.static_positional_weights[current_step]
            else:
                pass

            # seed tokens -> local clusters
            # 1. identify seed tokens
            if record_timing_breakdown:
                torch.cuda.synchronize(x.device)
                seed_start = time.perf_counter()
            conf_temp = confidence.clone()
            # conf_temp[~mask_token_mask] = -np.inf
            select_N = min((conf_temp > 0).sum().item(), select_N)    # if no masked tokens, exits
            if select_N == 0:
                GG = True
            else:
                seed_indices, select_N = self._select_seed_indices(conf_temp, select_N)
                if select_N == 0:
                    GG = True
            if record_timing_breakdown:
                torch.cuda.synchronize(x.device)
                seed_identification_time += time.perf_counter() - seed_start

            if not GG:
                if record_timing_breakdown:
                    torch.cuda.synchronize(x.device)
                    clusters_start = time.perf_counter()
                # 2. form clusters
                intervals = []
                exist_new_seed = False
                found_indices_in_memory = set()
                for b in range(seed_indices.shape[0]):
                    # determine from where to expansion: seed or previous interval
                    for interval_idx in range(select_N):
                        seed = seed_indices[b, interval_idx].item()

                        found_interval_in_memory = None
                        for idx, (start, end) in enumerate(memory_intervals):
                            if start <= seed <= end:
                                found_interval_in_memory = (start, end)  # (interval_idx, start, end)
                                found_indices_in_memory.add(idx)
                                break
                        if found_interval_in_memory is None:
                            assert x[b, seed] == self.mask_id
                            left, right = seed, seed
                            exist_new_seed = True
                        else:
                            left, right = found_interval_in_memory
                        # to left
                        n_consecutive_failures = 0
                        for i in range(left - 1, self.block_start - 1, -1):
                            if pre_demasked_index[b, i] or conf_temp[b, i] < self.exploration_threshold: # hit wall
                                n_consecutive_failures += 1
                                if n_consecutive_failures > self.tolerance_M:
                                    break
                            else:
                                n_consecutive_failures = 0
                                left = i
                        # to right
                        n_consecutive_failures = 0
                        for i in range(right + 1, self.block_end):
                            if pre_demasked_index[b, i] or conf_temp[b, i] < self.exploration_threshold: # hit wall
                                n_consecutive_failures += 1
                                if n_consecutive_failures > self.tolerance_M:
                                    break
                            else:
                                n_consecutive_failures = 0
                                right = i
                        # gathered: expanded interval in memory or new interval
                        intervals.append((left, right)) # [left, right]

                # merge intervals (gathered intervals + previous intervals in memory)
                old_intervals = [interval for idx, interval in enumerate(memory_intervals) if idx not in found_indices_in_memory]
                merged_intervals = self._merge_intervals(intervals + old_intervals)

                print(f"curr_step {current_step} [Divide] (block-{self.num_block} unmasked_ratio: {unmasked_ratio:.2f}) exp_step {exp_step + 1}: select_N = {select_N}, found intervals {merged_intervals}")

                # transfer tokens: "gentle" parallel decoding: select_N tokens are updated
                transfer_mask = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                transfer_mask.scatter_(dim=1, index=seed_indices, value=True)
                x[transfer_mask] = x0[transfer_mask]
                cache_session.on_tokens_updated(x, transfer_mask)

                # condition check for early convergence
                advance_threshold = 0.1
                exist_intervals_advance = (
                        exp_step == 0 or
                        len(merged_intervals) != len(memory_intervals) or
                        any((merged_intervals[i][1] - merged_intervals[i][0] + 1) / (memory_intervals[i][1] - memory_intervals[i][0] + 1) >= 1 + advance_threshold for i in range(len(merged_intervals)))
                )

                memory_intervals = merged_intervals
                record_intervals = merged_intervals

                # if converged (passive + positive), early stop and exits the Divide phase
                if exist_new_seed or exist_intervals_advance:
                    no_advance_n = 0
                else:
                    no_advance_n += 1
                    # consecutive failures
                    if no_advance_n >= 2:
                        print("====> no enough advancement, exit Divide phase.")
                        should_break = True
                intervals_mask = torch.zeros_like(x, dtype=torch.bool)
                for (start, end) in memory_intervals:
                    intervals_mask[:, start: end + 1] = True
                intervals_effective_mask = (intervals_mask & (x == self.mask_id))
                intervals_effective_density = confidence[intervals_effective_mask].mean().item() if intervals_effective_mask.sum().item() > 0 else 0.0
                if intervals_effective_density >= self.acceleration_low_threshold:
                    print(f"====> intervals density on masked tokens {intervals_effective_density:.2f} >= acceleration_low_threshold {self.acceleration_low_threshold}, exit Divide phase.")
                    should_break = True
                if record_timing_breakdown:
                    torch.cuda.synchronize(x.device)
                    clusters_formation_time += time.perf_counter() - clusters_start
            else:
                print(f"curr_step {current_step} [Divide] GG!!! (block-{self.num_block} unmasked_ratio: {unmasked_ratio:.2f}) exp_step {exp_step + 1}: select_N = {select_N}, with intervals {memory_intervals}")
                transfer_mask = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
            confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
            transfer_masks.append(transfer_mask.detach().cpu().numpy()[0][prompt_len:])
            history_intervals.append([(start - prompt_len, end - prompt_len) for start, end in record_intervals])

            if GG or should_break:
                break

            self.exp_N = select_N

        # 2. interval purification: rule out intervals such that have been fully demasked during Divide steps
        purified_intervals = []
        for start, end in memory_intervals:
            left, right = start, end
            # unmasked contraction
            while left <= right and x[0, left].item() != self.mask_id:
                left += 1
            while left <= right and x[0, right].item() != self.mask_id:
                right -= 1
            if left <= right:
                purified_intervals.append((left, right))
            # if (x[0, start:end + 1] == self.mask_id).any():
            #     purified_intervals.append((start, end))
        memory_intervals = purified_intervals

        print(f"=====> constructed decoding zones: {memory_intervals}")
        # for visualization
        history_intervals.append([(start - prompt_len, end - prompt_len) for start, end in memory_intervals])

        timing_breakdown = None
        if record_timing_breakdown:
            torch.cuda.synchronize(x.device)
            total_time = time.perf_counter() - phase_start
            timing_breakdown = {
                "total_time": total_time,
                "model_forward_time": model_forward_time,
                "seed_identification_time": seed_identification_time,
                "clusters_formation_time": clusters_formation_time,
                "other_time": total_time - model_forward_time - seed_identification_time - clusters_formation_time,
            }

        return x, memory_intervals, steps_used, outputs, confidences, transfer_masks, history_intervals, timing_breakdown

    def Conquer_phase(
            self,
            x: Tensor,
            prompt_mask: Tensor,
            cache_session,
            intervals: List[tuple[int, int]] = None,
            current_step: int = -1,
            record_timing_breakdown: bool = False,
    ):
        """
        Conquer Phase: do confidence-based parallel decoding on decoding zones.
            intervals: decoding zones from the Divide phase
        """
        steps_used = 0
        if not intervals:
            return x, steps_used, [], [], [], [], {
                "total_time": 0.0,
                "model_forward_time": 0.0,
                "boundary_adjustment_time": 0.0,
                "other_time": 0.0,
            } if record_timing_breakdown else None

        prompt_len = prompt_mask[0].sum().item()
        # interval_states = [{'coords': (start, end), 'status': 'active'} for start, end in intervals]

        outputs = []
        confidences = []
        transfer_masks = []
        history_intervals = []
        timing_breakdown = None
        if record_timing_breakdown:
            torch.cuda.synchronize(x.device)
            model_forward_time = 0.0
            boundary_adjustment_time = 0.0
            phase_start = time.perf_counter()

        # while any(s['status'] == 'active' for s in interval_states):
        dynamic_accel_mask = torch.zeros_like(x, dtype=torch.bool)
        for (start, end) in intervals:
            dynamic_accel_mask[:, start: end + 1] = True
        if dynamic_accel_mask.sum().item() > 0:
            while len(intervals) > 0:
                # only focus on decoding zones

                if record_timing_breakdown:
                    torch.cuda.synchronize(x.device)
                    forward_start = time.perf_counter()
                step = cache_session.forward_step(x)
                if record_timing_breakdown:
                    torch.cuda.synchronize(x.device)
                    model_forward_time += time.perf_counter() - forward_start
                    
                x0 = step.x0
                confidence = step.confidence
                current_step += 1
                steps_used += 1

                confidence[:, 0: self.block_start] = confidence[:, self.block_end:] = -np.inf   # semi support
                transfer_mask = confidence > 0.98   # extreme confidence updating is safe, echo Threorem 1.1
                # confidence_in_active_zones = torch.where(dynamic_accel_mask, confidence, -np.inf)

                # do confidence-base parallel decoding
                total_n_para_updated = 0
                total_n_cons_updated = 0
                for (itv_start, itv_end) in intervals:
                    # if state['status'] != 'active':
                    #     continue
                    # itv_start, itv_end = state['coords']
                    confidence_in_curr_zone = torch.zeros_like(x0, dtype=confidence.dtype, device=x0.device)
                    confidence_in_curr_zone[:, itv_start: itv_end + 1] = confidence[:, itv_start: itv_end + 1]
                    # strategy1: parallel decoding based on confidence
                    if self.acceleration_parallel_method == 'fixed':  # meaningless for Divide and Conquer
                        para_transfer_index = (confidence_in_curr_zone > self.acceleration_threshold)
                    elif self.acceleration_parallel_method == 'factor':
                        para_transfer_index = torch.zeros_like(confidence_in_curr_zone, dtype=torch.bool, device=x0.device)
                        for b in range(confidence_in_curr_zone.shape[0]):
                            conf_b = confidence_in_curr_zone[b].clone()
                            cand_mask = (conf_b > 0)  # (L,)
                            cand_idxs = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)  # (n,)
                            cand_confs = conf_b[cand_mask]  # (n,)
                            sorted_order = torch.argsort(cand_confs, descending=True)
                            cand_idxs = cand_idxs[sorted_order]
                            cand_confs = cand_confs[sorted_order]
                            for conf_idx, conf in reversed(list(enumerate(cand_confs.tolist()))):
                                para_feasible_n = int(self.acceleration_factor / (1 - conf + 1e-6) - 1)
                                if para_feasible_n >= conf_idx + 1:
                                    para_transfer_index.scatter_(dim=1, index=cand_idxs[:conf_idx + 1].unsqueeze(0), value=True)
                                    break
                    n_para_updated = para_transfer_index.sum().item()
                    transfer_mask |= para_transfer_index
                    total_n_para_updated += n_para_updated

                if total_n_para_updated == 0:
                    # final dance
                    cons_transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    _, topk_idxs = torch.topk(confidence, k=min(1, (confidence >= self.acceleration_low_threshold).sum().item()), dim=-1)  # (k,)
                    cons_transfer_index.scatter_(dim=1, index=topk_idxs, value=True)
                    total_n_cons_updated = cons_transfer_index.sum().item()
                    transfer_mask |= cons_transfer_index

                total_n_updated = total_n_para_updated + total_n_cons_updated

                x[transfer_mask] = x0[transfer_mask]
                cache_session.on_tokens_updated(x, transfer_mask)
                print(f"curr_step {current_step} [Conquer] (block-{self.num_block}): "
                      f"total_n_updated({total_n_updated}) = total_n_para_updated({total_n_para_updated})) + total_n_cons_updated({total_n_cons_updated})");

                # enrolling expansion
                if record_timing_breakdown:
                    torch.cuda.synchronize(x.device)
                    expansion_start = time.perf_counter()
                enrolled_intervals = []
                n_hit_tolerance = self.tolerance_M
                for i, itv in enumerate(intervals):
                    # if state['status'] == 'active':
                    # enroll left
                    start, end = itv

                    # potential enrolling expansion
                    left = start
                    left_search_min = intervals[i-1][1] + 1 if i > 0 else self.block_start
                    n_hit_wall = 0
                    for pos in range(start - 1, left_search_min - 1, -1):
                        if confidence[0, pos] <= self.acceleration_low_threshold:
                            n_hit_wall += 1
                            if n_hit_wall > n_hit_tolerance:
                                break
                        else:
                            n_hit_wall = 0
                            left = pos
                    right = end
                    right_search_max = intervals[i+1][0] - 1 if i < len(intervals) - 1 else self.block_end - 1
                    n_hit_wall = 0
                    for pos in range(right + 1, right_search_max):
                        if confidence[0, pos] <= self.acceleration_low_threshold:
                            n_hit_wall += 1
                            if n_hit_wall > n_hit_tolerance:
                                break
                        else:
                            n_hit_wall = 0
                            right = pos

                    # unmasked contraction
                    while left <= right and x[0, left].item() != self.mask_id:
                        left += 1
                    while left <= right and x[0, right].item() != self.mask_id:
                        right -= 1

                    if left <= right and (x[:, left: right + 1] == self.mask_id).any():
                       enrolled_intervals.append((left, right)) # keep active

                intervals = self._merge_intervals(enrolled_intervals)
                if record_timing_breakdown:
                    torch.cuda.synchronize(x.device)
                    boundary_adjustment_time += time.perf_counter() - expansion_start
                print(f"intervals after enrolling: {intervals}")

                # for visualization
                history_intervals.append([(start - prompt_len, end - prompt_len) for start, end in intervals])
                outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
                confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
                transfer_masks.append(transfer_mask.detach().cpu().numpy()[0][prompt_len:])

                # exit Conquer phase
                dynamic_accel_mask.fill_(False)
                for (start, end) in intervals:
                    dynamic_accel_mask[:, start: end + 1] = True
                if dynamic_accel_mask.sum().item() == 0:
                    print(f"=====> all zones comsumed or demasked, exit Conquer phase.")
                    break
                conf_rest_masked = confidence[dynamic_accel_mask & (x == self.mask_id)]
                dens_mean = conf_rest_masked.mean().item()
                print(f"=====> current confidence density in the rest zones: {dens_mean}")
                if dens_mean < self.acceleration_low_threshold:
                    print(f"=====> not enough confidence density in the rest zones ({dens_mean}), exit Conquer phase.")
                    break

        if record_timing_breakdown:
            torch.cuda.synchronize(x.device)
            total_time = time.perf_counter() - phase_start
            timing_breakdown = {
                "total_time": total_time,
                "model_forward_time": model_forward_time,
                "boundary_adjustment_time": boundary_adjustment_time,
                "other_time": total_time - model_forward_time - boundary_adjustment_time,
            }

        return x, steps_used, outputs, confidences, transfer_masks, history_intervals, timing_breakdown

    def Finalize_phase(
            self,
            x: Tensor,
            prompt_mask: Tensor,
            cache_session,
            current_step: int = -1,
            record_timing_breakdown: bool = False,
    ):
        """
            Finalize Phase: do margin-based parallel+top1 decoding on global context。
        """

        todo_steps = math.ceil((x[:, self.block_start: self.block_end] == self.mask_id).sum(dim=1).item() / self.mopup_speed)

        outputs = []
        confidences = []
        transfer_masks = []
        steps_used = 0
        if record_timing_breakdown:
            torch.cuda.synchronize(x.device)
            model_forward_time = 0.0
            phase_start = time.perf_counter()

        if todo_steps <= 0:
            return x, steps_used, outputs, confidences, transfer_masks, {
                "total_time": 0.0,
                "model_forward_time": 0.0,
                "other_time": 0.0,
            } if record_timing_breakdown else None

        prompt_len = prompt_mask[0].sum().item()
        num_masked = (x[:, self.block_start: self.block_end] == self.mask_id).sum().item()
        for i in range(todo_steps):
            if record_timing_breakdown:
                torch.cuda.synchronize(x.device)
                forward_start = time.perf_counter()
            step = cache_session.forward_step(x)
            if record_timing_breakdown:
                torch.cuda.synchronize(x.device)
                model_forward_time += time.perf_counter() - forward_start
            x0 = step.x0
            confidence = step.confidence
            logits = step.logits
            steps_used += 1

            confidence[:, 0: self.block_start] = confidence[:, self.block_end:] = -np.inf  # semi support

            # transfer_mask = conf_transfer_mask = confidence > 0.98   # extreme confidence updating is safe
            # n_conf_updated = conf_transfer_index.sum().item()

            # Use Margin update
            top2_logits = torch.topk(logits, k=2, dim=-1).values  # (b, l, 2)
            top2_margins = top2_logits[..., 0] - top2_logits[..., 1]  # (b, l)
            top2_margins[:, 0: self.block_start] = top2_margins[:, self.block_end:] = 0  # semi support
            #print statistics
            block_top2_margins = top2_margins[:, self.block_start: self.block_end]
            print(f"==> margins.mean={block_top2_margins.mean().item():.2f}, std={block_top2_margins.std().item():.2f}, "
                  f"max={block_top2_margins.max().item():.2f}, min={block_top2_margins.min().item():.2f}")
            top2_margins[x != self.mask_id] = 0

            transfer_index_margin = (top2_margins > self.mopup_margin_threshold) | (confidence > 0.98)

            # Test Fixed conf update
            # transfer_index_margin = confidence > self.acceleration_threshold
            # n_margin_updated = (transfer_index_margin & ~conf_transfer_index).sum().item()

            # n_margin_updated = 0
            n_margin_updated = transfer_index_margin.sum().item()
            n_topk_updated = min(num_masked, max(0, self.mopup_speed - n_margin_updated))
            if n_margin_updated > 0:
                transfer_mask = transfer_index_margin
            else:
                confidence[:, 0: self.block_start] = confidence[:, self.block_end:] = -np.inf  # semi support
                _, topk_idxs = torch.topk(confidence, k=n_topk_updated, dim=1)  # (b, l)
                transfer_mask = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                transfer_mask.scatter_(dim=1, index=topk_idxs, value=True)
            x[transfer_mask] = x0[transfer_mask]
            cache_session.on_tokens_updated(x, transfer_mask)
            num_masked = (x[:, self.block_start: self.block_end] == self.mask_id).sum().item()

            print(f"curr_step {current_step + i + 1} [Finalize] (block-{self.num_block} unmasked_ratio: {1 - num_masked/self.block_length:.2f}): n_updated({transfer_mask.sum().item()}) = n_margin_updated({n_margin_updated}) + n_topk_updated({n_topk_updated})")

            outputs.append(x0.detach().cpu().numpy()[0][prompt_len:])
            confidences.append(confidence.detach().cpu().to(torch.float32).numpy()[0][prompt_len:])
            transfer_masks.append(transfer_mask.detach().cpu().numpy()[0][prompt_len:])

            if num_masked == 0:
                break

        timing_breakdown = None
        if record_timing_breakdown:
            torch.cuda.synchronize(x.device)
            total_time = time.perf_counter() - phase_start
            timing_breakdown = {
                "total_time": total_time,
                "model_forward_time": model_forward_time,
                "other_time": total_time - model_forward_time,
            }

        return x, steps_used, outputs, confidences, transfer_masks, timing_breakdown

    @torch.no_grad()
    def generate(
        self,
        prompt,
        gen_length=256,
        max_steps=256,
        block_length=256,
        raw_queries=None,
        records=None
    ) -> GenerateOutput:
        """
        DiCo Controller
        """
        batch = prompt.shape[0]
        prompt_len = prompt.shape[1]
        assert batch == 1, "currently only support batch_size = 1"
        assert gen_length <= self.config.max_gen_length, f"gen_length must <= model_max_genlength({self.model_max_genlength})"
        assert max_steps <= self.config.max_steps, f"steps must <= model_max_steps({self.model_max_steps})"
        if records is None:
            records = ["metrics"]
        record_timing_breakdown = "timing_breakdown" in records

        # assert gen_length % block_length == 0
        # num_blocks = gen_length // block_length
        #
        # assert max_steps % num_blocks == 0

        outputs = []
        confidences = []
        transfer_masks = []
        phase_states = []  # [{'phase':'Divide/Conquer/Finalize', 'range': (start, end)}]
        history_intervals_all = []  # [{'inceptive_step': 0, 'history_intervals': [[(start, end), ...], [(start, end), ...], ...]}]
        accumulated_steps = 0
        start_time = time.perf_counter()

        # dynamic length
        adjusted_gen_lengths = self.length_strategy(
            self.model,
            prompt,
            self.config,
            gen_length,
            raw_queries=raw_queries,
        )  # (b,)
        n_blocks = (adjusted_gen_lengths.max().item() + block_length - 1) // block_length
        gen_length = n_blocks * block_length
        adjusted_steps = gen_length
        block_steps = adjusted_steps // n_blocks

        self.gen_length = gen_length
        self.max_steps = max_steps = adjusted_steps
        self.block_length = block_length
        self.block_steps = block_steps

        # initalize positional weights
        if self.positional_weights_type == 'static':
            self.static_positional_weights = self.precompute_static_positional_weights(
                gen_length=block_length, device=self.model.device, dtype=torch.float32
            )
        elif self.positional_weights_type != 'ratio':
            pass

        state = self.prepare_generation_state(
            prompt,
            gen_length,
            block_length,
            raw_queries=raw_queries,
            batch_gen_lengths=adjusted_gen_lengths,
            padded_gen_length=gen_length,
        )
        x = state.x
        prompt_mask = state.prompt_mask
        attention_mask = state.attention_mask
        total_lengths = state.total_lengths
        cache_session = self.cache_backend.start_session(self)
        timing_breakdown = None
        if record_timing_breakdown:
            timing_breakdown = {
                "summary": {
                    "total_time": 0.0,
                    "model_forward_time": 0.0,
                    "other_time": 0.0,
                    "dico_time": 0.0,
                },
                "Divide": {
                    "total_time": 0.0,
                    "model_forward_time": 0.0,
                    "seed_identification_time": 0.0,
                    "clusters_formation_time": 0.0,
                    "other_time": 0.0,
                },
                "Conquer": {
                    "total_time": 0.0,
                    "model_forward_time": 0.0,
                    "boundary_adjustment_time": 0.0,
                    "other_time": 0.0,
                },
                "Finalize": {
                    "total_time": 0.0,
                    "model_forward_time": 0.0,
                    "other_time": 0.0,
                },
            }

        for num_block in range(n_blocks):
            self.block_start = prompt_len + num_block * block_length
            self.block_end = prompt_len + (num_block + 1) * block_length
            self.num_block = num_block
            cache_session.start_block(
                x,
                self.block_start,
                self.block_end,
                total_lengths,
                prompt_mask,
                attention_mask=attention_mask,
            )

            for _ in range(int(block_steps * self.R_gate)):

                # ① Divide
                x, intervals, divide_steps, divide_outputs, divide_confidences, divide_transfer_masks, history_intervals, divide_timing_breakdown \
                    = self.Divide_phase(
                    x,
                    prompt_mask,
                    cache_session,
                    exp_steps=min(self.max_exploration_steps, max_steps - accumulated_steps),
                    current_step=accumulated_steps,
                    record_timing_breakdown=record_timing_breakdown,
                )

                # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Divide")
                outputs.extend(divide_outputs)
                confidences.extend(divide_confidences)
                transfer_masks.extend(divide_transfer_masks)
                phase_states.append(
                    {'phase': 'Divide', 'range': (accumulated_steps, accumulated_steps + divide_steps)})
                history_intervals_all.append({'inceptive_step': accumulated_steps, 'history_intervals': history_intervals})
                if record_timing_breakdown:
                    for key in timing_breakdown["Divide"]:
                        timing_breakdown["Divide"][key] += divide_timing_breakdown[key]
                # print(f"Divide phase ends, use steps: {divide_steps}, TPS: {(num_masked - num_masked_divide) / (divide_steps)}")
                # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Divide")
                accumulated_steps += divide_steps

                # ② Conquer
                if intervals:
                    x, conquer_steps, conquer_outputs, conquer_confidences, conquer_transfer_masks, history_intervals, conquer_timing_breakdown \
                        = self.Conquer_phase(
                            x,
                            prompt_mask,
                            cache_session,
                            intervals=intervals,
                            current_step=accumulated_steps,
                            record_timing_breakdown=record_timing_breakdown,
                    )

                    # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Conquer")
                    outputs.extend(conquer_outputs)
                    confidences.extend(conquer_confidences)
                    transfer_masks.extend(conquer_transfer_masks)
                    if record_timing_breakdown:
                        for key in timing_breakdown["Conquer"]:
                            timing_breakdown["Conquer"][key] += conquer_timing_breakdown[key]
                    phase_states.append(
                        {'phase': 'Conquer', 'range': (accumulated_steps, accumulated_steps + conquer_steps)})
                    history_intervals_all.append(
                        {'inceptive_step': accumulated_steps, 'history_intervals': history_intervals})
                    accumulated_steps += conquer_steps
                    # print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB; Conquer")

                    # check Finalize condition in the current block
                    masked_ratio = 1.0 * (x[:, self.block_start: self.block_end] == self.mask_id).sum().item() / block_length
                    if masked_ratio < (1 - self.R_gate):
                        print(
                            f"block {num_block}: D-C turn ends with unmased ratio: {(1 - masked_ratio) * 100}% (>{self.R_gate * 100}%)")
                        break
                else:
                    break

            # ③ Finalize
            x, finalize_steps, finalize_outputs, finalize_confidences, finalize_transfer_masks, finalize_timing_breakdown \
                = self.Finalize_phase(
                    x,
                    prompt_mask,
                    cache_session,
                    current_step=accumulated_steps,
                    record_timing_breakdown=record_timing_breakdown,
                )

            if finalize_steps > 0:
                outputs.extend(finalize_outputs)
                confidences.extend(finalize_confidences)
                transfer_masks.extend(finalize_transfer_masks)
                # print(f"Finalize phase ends, use steps: {finalize_steps}, TPS: {(num_masked_conquer - num_masked_finalize) / (finalize_steps)}")
                phase_states.append({'phase': 'Finalize', 'range': (accumulated_steps, accumulated_steps + finalize_steps)})
            else:
                pass
                # print(f"No Need for Finalize phase")
            if record_timing_breakdown:
                for key in timing_breakdown["Finalize"]:
                    timing_breakdown["Finalize"][key] += finalize_timing_breakdown[key]

            accumulated_steps += finalize_steps
            print(f"block {num_block} is decoded over in step {accumulated_steps}.")
            cache_session.finish_block()

        # compute recorder
        end_time = time.perf_counter()
        duration = end_time - start_time
        total_steps = accumulated_steps

        metrics = GenerationMetrics(
            use_seconds=duration,
            use_steps=total_steps,
            n_gen_tokens=gen_length,
            tokens_per_second=(gen_length / duration) if duration > 0 else 0,
            step_reduction_ratio=(gen_length / total_steps) if total_steps > 0 else 0
        )
        print(metrics)

        state_trace = {
            "outputs_all": outputs,
            "confidences_all": confidences,
            "transfer_masks_all": transfer_masks,
            "phase_states": phase_states,
            "history_intervals_all": history_intervals_all,
        }
        if record_timing_breakdown:
            timing_breakdown["summary"]["total_time"] = (
                timing_breakdown["Divide"]["total_time"]
                + timing_breakdown["Conquer"]["total_time"]
                + timing_breakdown["Finalize"]["total_time"]
            )
            timing_breakdown["summary"]["model_forward_time"] = (
                timing_breakdown["Divide"]["model_forward_time"]
                + timing_breakdown["Conquer"]["model_forward_time"]
                + timing_breakdown["Finalize"]["model_forward_time"]
            )
            timing_breakdown["summary"]["other_time"] = (
                timing_breakdown["Divide"]["other_time"]
                + timing_breakdown["Conquer"]["other_time"]
                + timing_breakdown["Finalize"]["other_time"]
            )
            timing_breakdown["summary"]["dico_time"] = (
                timing_breakdown["summary"]["total_time"]
                - timing_breakdown["summary"]["model_forward_time"]
                - timing_breakdown["summary"]["other_time"]
            )
            state_trace["timing_breakdown"] = timing_breakdown

        return GenerateOutput(
            out=x,
            state_trace=state_trace,
            metrics=metrics,
        )

def main():
    set_seed(1234)
    device = "cuda:0"

    # 4-shot prompt
    # few_shot_filename = "../prompts/gsm8k_shot.txt"
    # with open(few_shot_filename, "r", encoding="utf-8") as f:
    #     prompts= f.readlines()[0:3]

    # base gsm8k prompt
    gsm8k_dataset = load_dataset('openai/gsm8k', 'main')
    prompts = gsm8k_dataset['test']['question'][0:1]

    # base humaneval prompt
    # humaneval_dataset = load_dataset('openai/openai_humaneval')
    # prompts = humaneval_dataset['test']['prompt'][99:101]

    model_path = "/home/anyilin/works/dllm-research/models/LLaDA-8B-Instruct"

    # dream token info
    # model_path = "../models/Dream-7B-Instruct"
    # mask_id = 151666

    sampler = DiCo.build(
        model_path=model_path,
        device=device,
        config=DiCoConfig(
            exploration_seed_method="regular_interval"
        ),
        torch_dtype=torch.bfloat16,
        mask_id=126336,
    )
    # sampler.set_length_strategy(DAEDAL())
    # max_steps = 256
    # block_length = 64
    max_steps = 256
    block_lengthes = [max_steps]
    # exploration_thresholds = [0.15, 0.25, 0.4] # -> 0.25 is good for 'fixed', 'factor'
    prompt_prefix = ""

    for i, prompt_text in enumerate(prompts):
        print('=' * 20 + f" Generating prompt_idx: {i} " + "=" * 20)
        tokenizer = sampler.tokenizer
        prompt_text = prompt_prefix + prompt_text

        m = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)

        for block_length in block_lengthes:
            print('=' * 20 + f" block_length: {block_length} " + "=" * 20)
            OUT = sampler.generate(
                input_ids,
                gen_length=max_steps,
                max_steps=max_steps,
                block_length=block_length,
                records=["metrics", "timing_breakdown"]    
            )
            out = OUT.out
            ans = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            print(f"Prompt_{i}'s answer: {ans}\n")
            print(f"timing_breakdown: {OUT.state_trace['timing_breakdown']}\n")


if __name__ == '__main__':
    main()
