from __future__ import annotations
from typing import Any
from model_evaluation.core.errors import ConfigError

_ALLOWED_RUN_OVERRIDES={"offline","dataset_timeout_seconds","deployment","platform"}


def validate_run_overrides(run_spec: dict[str,Any]) -> None:
    overrides=run_spec.get('overrides') or {}
    if not isinstance(overrides,dict):
        raise ConfigError('run overrides must be an object')
    unknown=sorted(set(overrides)-_ALLOWED_RUN_OVERRIDES)
    if unknown:
        raise ConfigError(f'unsupported run override keys: {unknown}; supported={sorted(_ALLOWED_RUN_OVERRIDES)}')
    if 'offline' in overrides and not isinstance(overrides['offline'],bool):
        raise ConfigError('run override offline must be boolean')
    if 'dataset_timeout_seconds' in overrides:
        value=overrides['dataset_timeout_seconds']
        if isinstance(value,bool) or not isinstance(value,(int,float)) or value <= 0:
            raise ConfigError('run override dataset_timeout_seconds must be a positive number')
    if 'deployment' in overrides and not isinstance(overrides['deployment'],dict):
        raise ConfigError('run override deployment must be an object')
    if 'platform' in overrides and not isinstance(overrides['platform'],dict):
        raise ConfigError('run override platform must be an object')
