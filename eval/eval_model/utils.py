from dataclasses import fields


def build_config_from_kwargs(config_cls, kwargs):
    config_fields = {f.name for f in fields(config_cls)}
    config_kwargs = {
        key: kwargs[key]
        for key in config_fields
        if key in kwargs
    }
    return config_cls(**config_kwargs)
