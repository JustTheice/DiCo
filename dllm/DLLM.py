from abc import abstractmethod
from typing import List, Tuple, Literal, Dict
import pathlib

import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
# from visualizer import get_local

from dllm.tactics.cache_backend import CacheBackend, build_cache_backend
from dllm.tactics.cache_backend.none import NoCacheBackend
from dllm.tactics.length_strategy import VanillaLengthStrategy
from dllm.tactics.length_strategy.base import LengthStrategy
from dllm.tactics import build_length_strategy
from dllm.utils import add_gumbel_noise
from dataclasses import dataclass, fields, field

@dataclass
class DLLMConfig:
    # Model forward config
    cfg_scale: float = 0.0
    temperature: float = 0.0
    # Special token ids
    mask_id: int = 126336
    bos_id: int = 126080
    pad_id: int = 126081
    eos_id: int = 126081
    eot_id: int = 126348
    # Generation general config
    max_gen_length: int = 1024
    max_steps: int = 1024
    # Positional weight config
    positional_weights_type: Literal['ratio', 'static', 'none'] = 'none'
    weight_function_type: Literal['exponential', 'linear'] = 'exponential'
    max_weight: float = 1.0
    initial_min_weight: float = 0.05
    ur_factor: float = 1.0
    # dllm type
    dllm_type: Literal['llada', 'dream'] = 'llada'

@dataclass
class GenerationMetrics:
    use_seconds: float
    use_steps: int
    n_gen_tokens: int
    tokens_per_second: float
    step_reduction_ratio: float

@dataclass
class GenerateOutput:
    out: torch.Tensor
    metrics: GenerationMetrics
    state_trace: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    # def __post_init__(self):
    #     required_keys = {'outputs', 'confidences', 'transfer_masks'}
    #     if not required_keys.issubset(self.state_trace.keys()):
    #         missing = required_keys - self.state_trace.keys()
    #         raise ValueError(f"GenerateOutput.decoding_trace missing required keys: {missing}")

    # outputs: List[np.ndarray] = field(default_factory=list)
    # confidences: List[np.ndarray] = field(default_factory=list)
    # transfer_masks: List[np.ndarray] = field(default_factory=list)
    # phase_states: List= field(default_factory=list)
    # history_intervals_all: List = field(default_factory=list)


@dataclass
class ForwardStepOutput:
    x0: torch.Tensor
    confidence: torch.Tensor
    token_confidence: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor | None = None
    hidden_states: Tuple[torch.Tensor, ...] | None = None
    attentions: Tuple[torch.Tensor, ...] | None = None


@dataclass
class GenerationState:
    x: torch.Tensor
    attention_mask: torch.Tensor
    prompt_mask: torch.Tensor
    curr_decoding_pos: torch.Tensor
    mask_token_mask: torch.Tensor
    total_lengths: torch.Tensor
    batch_gen_lengths: torch.Tensor
    gen_length: int
    n_blocks: int
    prompt_len: int
    batch: int


class DLLM:
    """
        An Abstract Class for all dllms
    """
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            config: DLLMConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = model.device
        # binding configs to dllm
        for field in fields(config):
            print(f"{field.name}: {getattr(config, field.name)}")
            setattr(self, field.name, getattr(config, field.name))

        self.length_strategy = None
        self.cache_backend = None

    # Dynamic Generation Length. DAEDAL: https://doi.org/10.48550/arXiv.2508.00819
    def set_length_strategy(self, strategy):
        self.length_strategy = strategy

    def set_cache_backend(self, backend):
        self.cache_backend = backend

    @classmethod
    def build(
        cls,
        model_path: str,
        config: DLLMConfig | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        mask_id: int | None = None,
        token_overrides: dict | None = None,
        length_strategy: str | LengthStrategy = "vanilla",
        length_strategy_kwargs: dict | None = None,
        cache_backend: str | CacheBackend = "none",
    ):
        if config is None:
            config = DLLMConfig()

        token_overrides = token_overrides or {}
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        resolved_mask_id = mask_id if mask_id is not None else token_overrides.get("mask_id")
        assert resolved_mask_id is not None, "mask_id must be explicitly provided"

        config.mask_id = resolved_mask_id
        config.bos_id = token_overrides.get("bos_id", tokenizer.bos_token_id or config.bos_id)
        config.eos_id = token_overrides.get("eos_id", tokenizer.eos_token_id or config.eos_id)
        config.pad_id = token_overrides.get("pad_id", tokenizer.pad_token_id or config.pad_id)
        config.eot_id = token_overrides.get("eot_id", config.eos_id)

        sampler = cls.from_path(
            model_path=model_path,
            config=config,
            device=device,
            torch_dtype=torch_dtype,
        )
        if isinstance(length_strategy, str):
            sampler.set_length_strategy(build_length_strategy(length_strategy, **(length_strategy_kwargs or {})))
        else:
            sampler.set_length_strategy(length_strategy)
        if isinstance(cache_backend, str):
            sampler.set_cache_backend(build_cache_backend(cache_backend))
        else:
            sampler.set_cache_backend(cache_backend)
        return sampler

    @classmethod
    def from_path(
            cls,
            model_path: str,
            config: DLLMConfig,
            device: str | None = None,
            torch_dtype: torch.dtype = torch.bfloat16,
    ):
        model_name = pathlib.Path(model_path).name.lower()
        if model_name.startswith("llada"):
            config.dllm_type = 'llada'
        elif model_name.startswith("dream"):
            config.dllm_type = 'dream'
        print(f"Loading model and tokenizer from path: {model_path}, dllm_type: {config.dllm_type}")

        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
        # print(model)
        if device is not None:
            model.to(device=device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        return cls(model=model, tokenizer=tokenizer, config=config)

    @torch.no_grad()
    def model_forward(
            self,
            x: torch.Tensor,
            prompt_mask: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            **kwargs,
    ) -> tuple[object, torch.Tensor, tuple[torch.Tensor, ...] | None, tuple[torch.Tensor, ...] | None]:
        model_kwargs = dict(kwargs)
        if self.cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_mask] = self.mask_id
            x_ = torch.cat([x, un_x], dim=0)
            if attention_mask is not None:
                model_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask], dim=0)
            output = self.model(x_, **model_kwargs)
            logits_batch = output.logits
            logits, un_logits = torch.chunk(logits_batch, 2, dim=0)
            logits = un_logits + (self.cfg_scale + 1) * (logits - un_logits)
            hidden_states = None
            if model_kwargs.get("output_hidden_states") and output.hidden_states is not None:
                hidden_states = tuple(torch.chunk(h, 2, dim=0)[0] for h in output.hidden_states)
            attentions = None
            if model_kwargs.get("output_attentions") and output.attentions is not None:
                attentions = tuple(torch.chunk(a, 2, dim=0)[0] for a in output.attentions)
        else:
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask
            output = self.model(x, **model_kwargs)
            logits = output.logits
            hidden_states = output.hidden_states if model_kwargs.get("output_hidden_states") else None
            attentions = output.attentions if model_kwargs.get("output_attentions") else None

        if self.dllm_type == 'llada':
            pass
        elif self.dllm_type == 'dream':
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return output, logits, hidden_states, attentions

    @torch.no_grad()
    def forward_step(
            self,
            x: torch.Tensor,
            prompt_mask: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            output_hidden_states: bool = False,
            output_attentions: bool = False,
            return_probs: bool = False,
    ) -> ForwardStepOutput:
        # Deprecated compatibility path. New code should use cache_backend sessions.
        model_kwargs = {
            "output_hidden_states": output_hidden_states,
            "output_attentions": output_attentions,
        }
        if self.cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_mask] = self.mask_id
            x_ = torch.cat([x, un_x], dim=0)
            if attention_mask is not None:
                model_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask], dim=0)
            output = self.model(x_, **model_kwargs)
            logits_batch = output.logits
            logits, un_logits = torch.chunk(logits_batch, 2, dim=0)
            logits = un_logits + (self.cfg_scale + 1) * (logits - un_logits)
            hidden_states = None
            if output_hidden_states and output.hidden_states is not None:
                hidden_states = tuple(torch.chunk(h, 2, dim=0)[0] for h in output.hidden_states)
            attentions = None
            if output_attentions and output.attentions is not None:
                attentions = tuple(torch.chunk(a, 2, dim=0)[0] for a in output.attentions)
        else:
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask
            output = self.model(x, **model_kwargs)
            logits = output.logits
            hidden_states = output.hidden_states if output_hidden_states else None
            attentions = output.attentions if output_attentions else None

        if self.dllm_type == 'llada':
            pass
        elif self.dllm_type == 'dream':
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        logits_with_noise = add_gumbel_noise(logits, self.temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        probs = F.softmax(logits, dim=-1)
        token_confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        confidence = torch.where(x == self.mask_id, token_confidence, -np.inf)
        return ForwardStepOutput(
            x0=x0,
            confidence=confidence,
            token_confidence=token_confidence,
            logits=logits,
            probs=probs if return_probs else None,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def prepare_generation_state(
        self,
        prompt: torch.Tensor,
        gen_length: int,
        block_length: int,
        raw_queries: list[str] | None = None,
        batch_gen_lengths: torch.Tensor | None = None,
        padded_gen_length: int | None = None,
    ) -> GenerationState:
        batch = prompt.shape[0]
        prompt_len = prompt.shape[1]
        assert batch == 1, "currently only support batch_size = 1"

        if batch_gen_lengths is None:
            batch_gen_lengths = self.length_strategy(
                self.model,
                prompt,
                self.config,
                gen_length,
                raw_queries=raw_queries,
            )
        if padded_gen_length is None:
            gen_length = batch_gen_lengths.max().item()
        else:
            gen_length = padded_gen_length
        total_lengths = prompt_len + batch_gen_lengths
        n_blocks = (gen_length + block_length - 1) // block_length
        print(f"adjusted gen_length: {gen_length}, n_blocks: {n_blocks}.")

        x = torch.full(
            (batch, prompt_len + gen_length), self.eos_id, dtype=torch.long, device=self.model.device
        )
        x[:, :prompt_len] = prompt.clone()
        cols = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(batch, -1)
        mask_token_mask = (cols >= prompt_len) & (cols < total_lengths.unsqueeze(1))
        x[mask_token_mask] = self.mask_id
        attention_mask = cols < total_lengths.unsqueeze(1)
        prompt_mask = cols < prompt_len
        curr_decoding_pos = torch.full((batch,), prompt_len, dtype=torch.long, device=x.device)
        mask_token_mask = x == self.mask_id

        return GenerationState(
            x=x,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            curr_decoding_pos=curr_decoding_pos,
            mask_token_mask=mask_token_mask,
            total_lengths=total_lengths,
            batch_gen_lengths=batch_gen_lengths,
            gen_length=gen_length,
            n_blocks=n_blocks,
            prompt_len=prompt_len,
            batch=batch,
        )

    def build_block_mask(
        self,
        x: torch.Tensor,
        curr_decoding_pos: torch.Tensor,
        total_lengths: torch.Tensor,
        block_length: int,
    ) -> torch.Tensor:
        block_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for b in range(x.shape[0]):
            block_start = curr_decoding_pos[b]
            block_end = min(curr_decoding_pos[b] + block_length, total_lengths[b].item())
            block_mask[b, block_start:block_end] = True
        return block_mask

    def advance_decoding_position(
        self,
        x: torch.Tensor,
        curr_decoding_pos: torch.Tensor,
        total_lengths: torch.Tensor,
        block_length: int,
        mask_id: int,
    ) -> torch.Tensor:
        for b in range(x.shape[0]):
            if curr_decoding_pos[b] >= total_lengths[b]:
                continue
            block_start = curr_decoding_pos[b]
            block_end = min(curr_decoding_pos[b] + block_length, total_lengths[b].item())
            if (x[b, block_start:block_end] == mask_id).any():
                continue
            curr_decoding_pos[b] = block_end
        return curr_decoding_pos

    def _build_transfer_mask(
        self,
        confidences: torch.Tensor,
        effective_mask: torch.Tensor,
        logits: torch.Tensor | None = None,
        enable_fallback: bool = True,
    ) -> tuple[torch.Tensor, bool]:
        effective_confidences = confidences.masked_fill(~effective_mask, -np.inf)
        transfer_mask = torch.zeros_like(effective_mask, dtype=torch.bool, device=effective_mask.device)
        used_fallback = False
        for b in range(confidences.shape[0]):
            if self.decoding_method == "factor":
                # Fast-dllm: https://arxiv.org/abs/2505.22618
                conf_b = effective_confidences[b].clone()
                cand_mask = conf_b > 0
                cand_idxs = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)
                cand_confs = conf_b[cand_mask]
                sorted_order = torch.argsort(cand_confs, descending=True)
                cand_idxs = cand_idxs[sorted_order]
                cand_confs = cand_confs[sorted_order]
                for conf_idx, conf in reversed(list(enumerate(cand_confs.tolist()))):
                    para_feasible_n = int(self.factor / (1 - conf + 1e-6) - 1)
                    if para_feasible_n >= conf_idx + 1:
                        transfer_mask[b, cand_idxs[:conf_idx + 1]] = True
                        break
            elif self.decoding_method == "fixed":
                # Fast-dllm: https://arxiv.org/abs/2505.22618
                transfer_mask[b] = effective_confidences[b] > self.confidence_threshold
            elif self.decoding_method == "entropy_bound":
                # EB-Sampler: https://proceedings.neurips.cc/paper_files/paper/2025/hash/510e22f4a2da5212a64f1591736e2eaf-Abstract-Conference.html
                cand_mask = effective_confidences[b] > 0
                cand_idxs = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)
                if cand_idxs.numel() > 0:
                    cand_confs = effective_confidences[b, cand_idxs]
                    cand_idxs = cand_idxs[torch.argsort(cand_confs, descending=True)]
                    entropy = torch.distributions.Categorical(logits=logits[b, cand_idxs]).entropy()
                    acc_entropy = torch.cumsum(entropy, dim=0)
                    cummax_entropy = torch.cummax(entropy, dim=0).values
                    k = int((acc_entropy - cummax_entropy <= self.entropy_bound_gamma).sum().item())
                    if k > 0:
                        transfer_mask[b, cand_idxs[:k]] = True
            elif self.decoding_method == "topk":
                # Vanilla LLaDA/Dream
                if self.k <= 0:
                    raise ValueError("k must be a positive integer.")
                n_effective = (effective_confidences[b] > 0).sum().item()
                if n_effective > 0:
                    _, select_index = torch.topk(effective_confidences[b], k=min(self.k, n_effective))
                    transfer_mask[b, select_index] = True

            # top-1 fallback
            if enable_fallback and not transfer_mask[b].any() and effective_mask[b].any():
                _, select_index = torch.topk(effective_confidences[b], k=1)
                transfer_mask[b, select_index] = True
                used_fallback = True
        return transfer_mask, used_fallback

    @torch.no_grad()
    @abstractmethod
    def generate(
        self,
        prompt,
        gen_length=256,
        max_steps=256,
        block_length=256,
        raw_queries=None,
        record=None,
        **kwargs
    ) -> GenerateOutput:
        pass

    def precompute_static_positional_weights(
        self,
        gen_length: int,
        device: torch.device = 'cuda',
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
            precompute a weight matrix shaped (gen_length,), static weights for all steps
        """
        assert gen_length > 0, "gen_length must > 0"
        max_weight = self.max_weight
        min_weight = self.initial_min_weight

        if gen_length == 1:
            return torch.full((gen_length,), max_weight, device=device, dtype=dtype)
        positions = torch.arange(gen_length, device=device, dtype=dtype)  # (gen_length,)
        # compute lambda_decay
        lambda_decay = -torch.log(torch.tensor(min_weight)) / (gen_length - 1)  # scalar
        # compute step_positional_weights
        positional_weights = torch.exp(-lambda_decay * positions)  # (max_steps, gen_length)
        return positional_weights

    def compute_dynamic_positional_weights(
        self,
        gen_length: int,
        unmasked_ratio: float,
        device: torch.device = 'cuda',
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
            compute a weight matrix shaped (gen_length,) for the current step based on unmasked_ratio
        """
        max_weight = self.max_weight
        initial_min_weight = self.initial_min_weight

        assert gen_length >= 0, f"gen_length={gen_length}, it must > 0"
        if gen_length == 1:
            return torch.full((gen_length,), max_weight, device=device, dtype=dtype)

        positions = torch.arange(gen_length, device=device, dtype=dtype)  # (gen_length, )

        # compute min_weight based on unmasked_ratio
        min_weight = min(1.0, initial_min_weight + self.ur_factor * unmasked_ratio)
        if self.weight_function_type == "linear":
            return torch.linspace(max_weight, min_weight, gen_length, device=device, dtype=dtype)
        if self.weight_function_type == "exponential":
            assert min_weight > 0
            lambda_decay = -torch.log(torch.tensor(min_weight / max_weight, device=device, dtype=dtype)) / (gen_length - 1)
            return max_weight * torch.exp(-lambda_decay * positions)
        raise ValueError(f"Unsupported weight_function_type: {self.weight_function_type}")
