from __future__ import annotations
import json, math
from typing import Any


def reject_non_finite(value: Any, path: str='<root>') -> None:
    if isinstance(value,float) and not math.isfinite(value):
        raise ValueError(f'non-finite number is forbidden at {path}')
    if isinstance(value,dict):
        for k,v in value.items(): reject_non_finite(v,f'{path}.{k}')
    elif isinstance(value,list):
        for i,v in enumerate(value): reject_non_finite(v,f'{path}[{i}]')


def _pairs(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise ValueError(f'duplicate JSON key: {k}')
        out[k]=v
    return out


def loads(text: str):
    obj=json.loads(text,object_pairs_hook=_pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'non-finite JSON number: {x}')))
    reject_non_finite(obj); return obj


def dumps(obj: Any, **kwargs) -> str:
    reject_non_finite(obj); return json.dumps(obj,allow_nan=False,**kwargs)
