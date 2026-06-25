import random

import numpy as np
import torch

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_token_mask, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA-8B-Instruct employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_token_mask.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_token_mask.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def decode_outputs(output_ids, tokenizer):
    """
        解码token_ids -> tokens
        output_ids: list of steps*gen_length
    """
    decoded_outputs = [
        [tokenizer.decode(token_id) for token_id in output_each_step]
        for output_each_step in output_ids
    ]
    return decoded_outputs


def _precompute_positional_weights(
    max_steps: int,
    gen_length: int,
    max_weight: float,
    initial_min_weight: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
        precompute a weight matrix shaped (max_steps, gen_length)
        每一行代表一个step的权重曲线，曲线随step增加而变得平缓。
    """
    assert gen_length > 0 and max_steps > 0, "gen_length and max_steps must > 0"
    if gen_length == 1:
        return torch.full((max_steps, gen_length), max_weight, device=device, dtype=dtype)

    positions = torch.arange(gen_length, device=device, dtype=dtype).unsqueeze(0)  # (1, gen_length)
    if max_steps == 1:
        lambda_decay = -torch.log(torch.tensor(initial_min_weight, device=device, dtype=dtype)) / (gen_length - 1)
        return torch.exp(-lambda_decay * positions)

    # compute positional weights
    steps = torch.arange(max_steps, device=device, dtype=torch.float32).unsqueeze(1)  # (max_steps, 1)
    # compute min_weight on each step via linear interpolation
    t = steps / (max_steps - 1)  # interpolation factor
    min_weights = initial_min_weight + (max_weight - initial_min_weight) * t  # (max_steps, 1)
    # compute lambda_decay on each step, according to t
    lambda_decays = -torch.log(min_weights) / (gen_length - 1)  # (max_steps, 1)
    # compute step_position_weights via broadcasting
    step_position_weights = torch.exp(-lambda_decays * positions)  # (max_steps, gen_length)

    return step_position_weights

def compute_dynamic_positional_weights(
    gen_length: int,
    unmasked_ratio: float,
    weight_function_type: str = "exponential",
    max_weight: float = 1.0,
    initial_min_weight: float = 0.1,
    ur_factor: float = 1.0,
    device: torch.device = 'cuda',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
        compute a weight matrix shaped (gen_length,) for the current step based on unmasked_ratio
    """

    if gen_length == 1:
        return torch.full((gen_length,), max_weight, device=device, dtype=dtype)

    positions = torch.arange(gen_length, device=device, dtype=dtype)

    # compute min_weight based on unmasked_ratio
    min_weight = min(1.0, initial_min_weight + ur_factor * unmasked_ratio)
    if weight_function_type == "linear":
        return torch.linspace(max_weight, min_weight, gen_length, device=device, dtype=dtype)
    if weight_function_type == "exponential":
        lambda_decay = -torch.log(torch.tensor(min_weight / max_weight, device=device, dtype=dtype)) / (gen_length - 1)
        return max_weight * torch.exp(-lambda_decay * positions)
    raise ValueError(f"Unsupported weight_function_type: {weight_function_type}")


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
