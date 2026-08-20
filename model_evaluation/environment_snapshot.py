from __future__ import annotations

import json
import platform
import re
import sys
from importlib import metadata


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def controller_environment_snapshot() -> dict:
    packages: dict[str, dict] = {}
    for distribution in metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = _canonical_name(raw_name)
        record = {"name": name, "version": distribution.version}
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            try:
                record["direct_url"] = json.loads(direct_url)
            except json.JSONDecodeError:
                record["direct_url_unparsed"] = direct_url.strip()[:2000]
        packages[name] = record
    return {
        "schema_version": "1.0",
        "scope": "controller-python-environment",
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": [packages[name] for name in sorted(packages)],
    }


def requirements_lock_text(snapshot: dict) -> str:
    lines = [
        "# Exact controller environment captured by eval-manager environment-snapshot.",
        "# Backend and evaluator environments are separate execution roles and need",
        "# their own lock/export when they use different Python environments.",
    ]
    lines.extend(
        f"{package['name']}=={package['version']}" for package in snapshot["packages"]
    )
    return "\n".join(lines) + "\n"
