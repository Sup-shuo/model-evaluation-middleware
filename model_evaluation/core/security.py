from __future__ import annotations

import os
import re
from collections.abc import Mapping

# Adapter RPC is a trust boundary. Core intentionally knows only generic process
# variables here. Vendor/runtime installation hints belong to Platform-owned
# adapter parameters and are passed as data, not as Core hard-coded env names.
_SAFE_NAMES = {
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE",
    "TZ", "TMPDIR", "TEMP", "TMP", "PYTHONIOENCODING", "PYTHONUTF8",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "LD_LIBRARY_PATH",
}
_SECRET_NAME_RE = re.compile(
    r"(?:^|_)(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION|PRIVATE_?KEY)(?:_|$)",
    re.IGNORECASE,
)


def looks_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME_RE.search(name))


def adapter_subprocess_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    src = dict(os.environ if source is None else source)
    env: dict[str, str] = {}
    for name in _SAFE_NAMES:
        value = src.get(name)
        if value is not None and not looks_secret_name(name):
            env[name] = value
    env.setdefault("LANG", "C.UTF-8")
    return env


def execution_subprocess_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Minimal inherited environment for managed backend/evaluator processes.

    Runtime/device-specific state is supplied through resolved EnvPatch objects;
    secrets are injected only through explicit secret references.
    """
    return adapter_subprocess_env(source)


def redact_text(text: str, secret_values: list[str] | tuple[str, ...] | None = None) -> str:
    out = text
    for value in secret_values or ():
        if value:
            out = out.replace(value, "<redacted>")
    return out


def redact_diagnostic(value, secret_values: list[str] | tuple[str, ...] | None = None):
    """Recursively redact resolved secret values from user-visible diagnostics."""
    secrets = tuple(secret_values or ())
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, dict):
        return {str(k): redact_diagnostic(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_diagnostic(v, secrets) for v in value]
    return value
