from __future__ import annotations

from .dkv import DKVCacheBackend
from .fastdllm import FastDLLMCacheBackend
from .none import NoCacheBackend


def build_cache_backend(name: str, **kwargs) -> NoCacheBackend | FastDLLMCacheBackend | DKVCacheBackend:
    name = name.lower()
    if name == "none":
        return NoCacheBackend()
    if name == "fastdllm-prefix":
        return FastDLLMCacheBackend(mode="prefix")
    if name == "fastdllm-dual":
        return FastDLLMCacheBackend(mode="dual")
    if name == "dkv-decode":
        return DKVCacheBackend(mode="decode")
    if name == "dkv-prefix-decode":
        return DKVCacheBackend(mode="prefix-decode")
    raise ValueError(f"Unknown cache backend: {name}")
