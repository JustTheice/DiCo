import inspect
from dataclasses import asdict, is_dataclass

from dllm.tactics.length_strategy.base import LengthStrategy
from dllm.tactics.length_strategy.daedal import DAEDALLengthStrategy
from dllm.tactics.length_strategy.vanilla import VanillaLengthStrategy


def build_length_strategy(name: str, **kwargs) -> LengthStrategy:
    normalized_name = name.lower()
    if normalized_name == "vanilla":
        return VanillaLengthStrategy()
    if normalized_name == "daedal":
        return DAEDALLengthStrategy(**kwargs)
    raise ValueError(f"Unknown length strategy: {name}")


def extract_length_strategy_kwargs(strategy_name: str, source: dict | object, device: str | None = None) -> dict:
    strategy_name = strategy_name.lower()
    if strategy_name == "vanilla":
        return {}

    if strategy_name == "daedal":
        strategy_cls = DAEDALLengthStrategy
    else:
        raise ValueError(f"Unknown length strategy: {strategy_name}")

    if isinstance(source, dict):
        values = source
    elif is_dataclass(source):
        values = asdict(source)
    else:
        values = vars(source)

    kwargs = {
        name: values[name]
        for name in inspect.signature(strategy_cls.__init__).parameters.keys()
        if name not in {"self", "device"} and name in values
    }
    return kwargs
