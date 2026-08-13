from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from model_evaluation.core.errors import CompatibilityError

@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    reasons: list[str]
    optional_misses: list[str]

    @property
    def status(self) -> str:
        return "compatible" if self.compatible else "incompatible"


def _numeric_version(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "none", "n/a", "na"}:
        return None
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)(?:[-+][0-9A-Za-z.-]+)?", text)
    if not match:
        return None
    return tuple(int(x) for x in match.group(1).split("."))


def _version_ge(actual: Any, expected: Any) -> bool:
    left = _numeric_version(actual); right = _numeric_version(expected)
    if left is None or right is None:
        return False
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (width - len(right))


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "equals": return actual == expected
    if op == "in":
        if not isinstance(expected, list): return False
        return actual in expected
    if op == "gte":
        try: return actual >= expected
        except TypeError: return False
    if op == "lte":
        try: return actual <= expected
        except TypeError: return False
    if op == "min_version": return _version_ge(actual, expected)
    if op == "exists": return actual is not None
    raise CompatibilityError(f"unsupported requirement operator: {op}")


def evaluate(requirement_set: dict, facts: dict[str, Any]) -> CompatibilityReport:
    reasons: list[str] = []
    optional: list[str] = []
    for req in requirement_set.get("requirements", []):
        path = req["path"]
        actual = facts.get(path)
        ok = _compare(actual, req["op"], req.get("value"))
        if ok:
            continue
        msg = req.get("message") or f"requirement failed: {path} {req['op']} {req.get('value')!r}; actual={actual!r}"
        if req.get("optional"):
            optional.append(msg)
        else:
            reasons.append(msg)
    return CompatibilityReport(not reasons, reasons, optional)


def merge_fact_sets(*sets: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for facts in sets:
        for key, value in facts.items():
            if key in out and out[key] != value:
                raise CompatibilityError(f"conflicting canonical fact {key}: {out[key]!r} != {value!r}")
            out[key] = value
    return out


def _merge_capabilities(facts: dict[str, Any], desc: dict, *, label: str, namespace: str) -> dict[str, Any]:
    prefix=namespace+'.'
    for key, value in (desc.get('capabilities', {}).get('values', {}) or {}).items():
        if not isinstance(key,str) or not key.startswith(prefix):
            raise CompatibilityError(f'{label} capability escapes owned namespace {prefix!r}: {key!r}')
        if key in facts:
            if facts[key] != value:
                raise CompatibilityError(
                    f"{label} capability attempts to override canonical fact {key}: {facts[key]!r} -> {value!r}"
                )
            continue
        facts[key] = value
    return facts


def facts_from_device(desc: dict) -> dict[str, Any]:
    facts = {
        "device.vendor": desc.get("vendor"),
        "device.type": desc.get("device_type"),
        "device.count": len(desc.get("devices", [])),
    }
    return _merge_capabilities(facts, desc, label='device', namespace='device')


def facts_from_runtime(desc: dict) -> dict[str, Any]:
    facts = {
        "runtime.family": desc.get("family"),
        "runtime.version": desc.get("version"),
        "runtime.available": desc.get("available"),
    }
    return _merge_capabilities(facts, desc, label='runtime', namespace='runtime')


def facts_from_environment(desc: dict, prefix: str = 'environment') -> dict[str, Any]:
    facts = {f'{prefix}.provider': desc.get('provider'), f'{prefix}.identity': desc.get('identity')}
    caps = desc.get('capabilities', {}).get('values', {}) or {}
    for key, value in caps.items():
        if not isinstance(key,str) or not key.startswith('environment.'):
            raise CompatibilityError(f"environment capability escapes owned namespace 'environment.': {key!r}")
        canonical=f"{prefix}.{key.removeprefix('environment.')}"
        if canonical in facts and facts[canonical] != value:
            raise CompatibilityError(f'environment capability attempts to override canonical fact {canonical}')
        facts[canonical]=value
    return facts


def facts_from_service(desc: dict) -> dict[str, Any]:
    caps = desc.get('capabilities', {}).get('values', {}) or {}
    protocols=desc.get('protocols', {}) or {}
    tokenizer=desc.get('tokenizer') or {}; local_tokenizer=bool(tokenizer.get('mode')=='local' and tokenizer.get('path'))
    remote_tokenizer=bool(caps.get('service.tokenize') and caps.get('service.detokenize') and all(k in protocols for k in ('tokenizer_info','tokenize','detokenize')))
    facts = {
        'service.type': desc.get('service_type'),
        'service.ownership': desc.get('ownership'),
        'service.context_length': desc.get('context_length'),
        'service.auth_mode': (desc.get('auth') or {}).get('mode'),
        'service.local_tokenizer': local_tokenizer,
        'service.remote_tokenizer': remote_tokenizer,
        'service.tokenizer_available': local_tokenizer or remote_tokenizer,
    }
    for name in protocols:
        facts[f'service.protocol.{name}'] = True
    _merge_capabilities(facts, desc, label='service', namespace='service')
    return facts


def device_runtime_compatibility(device: dict, runtime: dict) -> CompatibilityReport:
    vendor = device.get("vendor")
    allowed = (runtime.get("capabilities", {}).get("values", {}) or {}).get("runtime.compatible_device_vendors")
    if allowed is None:
        return CompatibilityReport(True, [], ["runtime did not declare compatible device vendors"])
    if not isinstance(allowed, list):
        return CompatibilityReport(False, ["runtime.compatible_device_vendors must be an array"], [])
    if vendor not in allowed:
        return CompatibilityReport(False, [f"device/runtime mismatch: vendor={vendor!r} not in runtime compatible vendors {allowed!r}"], [])
    return CompatibilityReport(True, [], [])
