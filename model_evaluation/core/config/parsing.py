from __future__ import annotations
from typing import Any
import yaml
from model_evaluation.core.serialization import json_loads_strict, reject_non_finite

class UniqueKeySafeLoader(yaml.SafeLoader):
    pass

def _construct_mapping(loader: UniqueKeySafeLoader, node, deep=False):
    loader.flatten_mapping(node)
    mapping={}
    for key_node,value_node in node.value:
        key=loader.construct_object(key_node,deep=deep)
        if key in mapping: raise ValueError(f'duplicate YAML key: {key!r}')
        mapping[key]=loader.construct_object(value_node,deep=deep)
    return mapping

UniqueKeySafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,_construct_mapping)

def load_json_strict(text: str) -> Any:
    return json_loads_strict(text)

def load_yaml_strict(text: str) -> Any:
    obj=yaml.load(text,Loader=UniqueKeySafeLoader)
    reject_non_finite(obj)
    return obj
