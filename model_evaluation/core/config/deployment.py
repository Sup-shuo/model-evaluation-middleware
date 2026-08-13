from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from model_evaluation.core.errors import ConfigError

_TOKEN = re.compile(r"\{([A-Za-z0-9_.]+)\}")
_ALLOWED_PREFIXES = ('model.','platform.')


def _lookup(context: dict[str, Any], dotted: str) -> Any:
    prefix=next((p for p in _ALLOWED_PREFIXES if dotted.startswith(p)),None)
    if prefix is None:
        raise ConfigError(f"deployment template placeholder must start with one of {_ALLOWED_PREFIXES}: {dotted}")
    root_key=prefix[:-1]; cur: Any = context.get(root_key)
    for part in dotted.split('.')[1:]:
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"deployment template placeholder is unavailable: {dotted}")
        cur = cur[part]
    if cur is None:
        raise ConfigError(f"deployment template placeholder resolves to null: {dotted}")
    if isinstance(cur, (dict, list)):
        raise ConfigError(f"deployment template placeholder must resolve to a scalar: {dotted}")
    return cur


def render_deployment_template(template: str, *, model: dict[str, Any], platform: dict[str, Any] | None=None) -> str:
    if not isinstance(template, str) or not template:
        raise ConfigError("deployment path template must be a non-empty string")
    used = False

    def repl(match: re.Match[str]) -> str:
        nonlocal used
        used = True
        return str(_lookup({'model':model,'platform':platform or {}}, match.group(1)))

    rendered = _TOKEN.sub(repl, template)
    if '{' in rendered or '}' in rendered:
        raise ConfigError(f"unsupported/unbalanced deployment template expression: {template}")
    if not used:
        raise ConfigError(f"deployment template contains no supported model placeholder: {template}")
    return rendered


def _resolve_under_root(root_value: str | None, rendered: str, *, label: str) -> str:
    raw = Path(rendered)
    if root_value is None:
        if not raw.is_absolute():
            raise ConfigError(f"{label} resolves to a relative path without an absolute root: {rendered}")
        return str(raw.resolve())
    root = Path(str(root_value))
    if not root.is_absolute():
        raise ConfigError(f"{label} root must be absolute: {root}")
    root = root.resolve()
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if candidate != root and root not in candidate.parents:
        raise ConfigError(f"{label} escapes configured root: root={root} resolved={candidate}")
    return str(candidate)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out=copy.deepcopy(base)
    for key,value in patch.items():
        if isinstance(value,dict) and isinstance(out.get(key),dict): out[key]=_deep_merge(out[key],value)
        else: out[key]=copy.deepcopy(value)
    return out


def resolve_deployment_profile(deployment: dict[str, Any], model: dict[str, Any], platform: dict[str, Any] | None=None, deployment_override: dict[str, Any] | None=None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve model-dependent deployment fields without embedding backend logic.

    Supported model_location forms:
      - local_path: explicit absolute path
      - root/root_template + path_template: templates may reference model.* or platform.* scalar fields
      - local ModelSpec source.ref: used when source.type=local and no path is set

    tokenizer_template may be resolved under tokenizer_root (or root when omitted).
    When a ModelSpec carries a logical tokenizer.ref and the deployment provides
    a model root, the reference is resolved under that machine-owned root.  This
    keeps catalog entries portable while ensuring managed backends receive an
    absolute local tokenizer path.
    """
    effective = copy.deepcopy(deployment)
    if deployment_override:
        if not isinstance(deployment_override,dict): raise ConfigError('run override deployment must be an object')
        forbidden=set(deployment_override)-{'parameters','endpoint','model_location'}
        if forbidden: raise ConfigError(f'run deployment override cannot change structural fields: {sorted(forbidden)}')
        effective=_deep_merge(effective,deployment_override)
    location = copy.deepcopy(effective.get('model_location') or {})
    if not location:
        return effective, {"mode": "none", "override_applied": bool(deployment_override)}

    explicit = location.get('local_path')
    template = location.get('path_template')
    root_value=location.get('root')
    root_template=location.get('root_template')
    if root_value and root_template: raise ConfigError('deployment model_location cannot set both root and root_template')
    if root_template: root_value=render_deployment_template(str(root_template),model=model,platform=platform)
    if explicit and template:
        raise ConfigError("deployment model_location cannot set both local_path and path_template")

    resolution: dict[str, Any] = {"mode": "explicit" if explicit else "none", "override_applied": bool(deployment_override)}
    if template:
        rendered = render_deployment_template(str(template),model=model,platform=platform)
        local_path = _resolve_under_root(root_value, rendered, label='model_location.path_template')
        location['local_path'] = local_path
        resolution = {
            "mode": "template",
            "path_template": str(template),
            "root": root_value,"root_template": root_template,
            "resolved_local_path": local_path,
            "override_applied": bool(deployment_override),
        }
    elif not explicit and (model.get('source') or {}).get('type') == 'local':
        ref = str((model.get('source') or {}).get('ref') or '')
        if not Path(ref).is_absolute():
            raise ConfigError("ModelSpec source.type=local requires an absolute source.ref when deployment has no local_path/template")
        location['local_path'] = str(Path(ref).resolve())
        resolution = {"mode": "model_source", "resolved_local_path": location['local_path'], "override_applied": bool(deployment_override)}

    tokenizer_template = location.get('tokenizer_template')
    if tokenizer_template and location.get('tokenizer_path'):
        raise ConfigError("deployment model_location cannot set both tokenizer_path and tokenizer_template")
    if tokenizer_template:
        rendered = render_deployment_template(str(tokenizer_template),model=model,platform=platform)
        tokenizer_root = location.get('tokenizer_root', root_value)
        tokenizer_root_template=location.get('tokenizer_root_template')
        if location.get('tokenizer_root') and tokenizer_root_template:
            raise ConfigError('deployment model_location cannot set both tokenizer_root and tokenizer_root_template')
        if tokenizer_root_template:
            tokenizer_root=render_deployment_template(str(tokenizer_root_template),model=model,platform=platform)
        tokenizer_path = _resolve_under_root(tokenizer_root, rendered, label='model_location.tokenizer_template')
        location['tokenizer_path'] = tokenizer_path
        resolution['tokenizer_template'] = str(tokenizer_template)
        resolution['resolved_tokenizer_path'] = tokenizer_path
    elif not location.get('tokenizer_path'):
        tokenizer = model.get('tokenizer') or {}
        tokenizer_ref = tokenizer.get('ref') if isinstance(tokenizer, dict) else None
        if tokenizer_ref is not None:
            if not isinstance(tokenizer_ref, str) or not tokenizer_ref.strip():
                raise ConfigError("ModelSpec tokenizer.ref must be a non-empty string")
            tokenizer_ref = tokenizer_ref.strip()
            raw_tokenizer = Path(tokenizer_ref)
            # Preserve an explicitly absolute tokenizer path for backwards
            # compatibility.  Only logical (relative) refs are machine-bound
            # through the System-owned model root.
            if raw_tokenizer.is_absolute():
                tokenizer_path = str(raw_tokenizer.resolve())
            elif root_value is not None:
                tokenizer_path = _resolve_under_root(
                    root_value,
                    tokenizer_ref,
                    label='model.tokenizer.ref',
                )
            else:
                tokenizer_path = None
            if tokenizer_path is not None:
                location['tokenizer_path'] = tokenizer_path
                resolution['tokenizer_ref'] = tokenizer_ref
                resolution['resolved_tokenizer_path'] = tokenizer_path

    # Template control fields are planning inputs, not backend-facing location fields.
    for key in ('root', 'root_template', 'path_template', 'tokenizer_root', 'tokenizer_root_template', 'tokenizer_template'):
        location.pop(key, None)
    effective['model_location'] = location
    return effective, resolution
