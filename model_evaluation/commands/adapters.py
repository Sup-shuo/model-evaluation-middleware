from __future__ import annotations

from pathlib import Path

from model_evaluation.core.errors import ConfigError
from model_evaluation.core.registry.adapter_registry import AdapterRegistry


def check_adapter_root(value: str | Path, schemas) -> dict:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ConfigError(f"adapter root is not a directory: {root}")
    registry = AdapterRegistry(root, schemas, isolated_root=True)
    identities = registry.identities()
    if not identities:
        raise ConfigError(
            f"adapter root contains no executable <kind>/<name>/adapter launchers: {root}"
        )
    return {
        "ok": True,
        "root": str(root),
        "count": len(identities),
        "adapters": [
            {
                "kind": identity.kind,
                "name": identity.name,
                "version": identity.version,
                "entry": str(identity.path),
                "operations": list(identity.manifest.get("operations") or []),
            }
            for identity in identities
        ],
    }
