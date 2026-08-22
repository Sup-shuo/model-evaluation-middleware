from __future__ import annotations

from typing import Any


# Stable public terms, not a closed allow-list. Adapters may publish namespaced
# extension capabilities without editing Core.
CORE_CAPABILITIES: dict[str, dict[str, str]] = {
    "device.vendor": {"value_type": "string", "description": "Accelerator vendor."},
    "device.type": {"value_type": "string", "description": "Accelerator class."},
    "device.count": {"value_type": "integer", "description": "Visible device count."},
    "runtime.family": {"value_type": "string", "description": "Runtime family."},
    "runtime.version": {"value_type": "string", "description": "Runtime version."},
    "runtime.available": {"value_type": "boolean", "description": "Runtime availability."},
    "service.type": {"value_type": "string", "description": "Inference service type."},
    "service.ownership": {"value_type": "string", "description": "Service ownership mode."},
    "service.context_length": {"value_type": "integer", "description": "Service context limit."},
    "service.auth_mode": {"value_type": "string", "description": "Service authentication mode."},
    "service.local_tokenizer": {"value_type": "boolean", "description": "Local tokenizer availability."},
    "service.remote_tokenizer": {"value_type": "boolean", "description": "Remote tokenizer availability."},
    "service.tokenizer_available": {"value_type": "boolean", "description": "Any usable tokenizer."},
}


def vocabulary_scope(path: str) -> str:
    """Classify a capability without rejecting unknown extension names."""

    return "core" if path in CORE_CAPABILITIES else "extension"


def requirement_diagnostic(
    requirement: dict[str, Any],
    *,
    actual: Any,
    message: str,
) -> dict[str, Any]:
    diagnostic = {
        "code": "CAPABILITY_REQUIREMENT_FAILED",
        "severity": "warning" if requirement.get("optional") else "error",
        "path": str(requirement["path"]),
        "operator": str(requirement["op"]),
        "actual": actual,
        "optional": bool(requirement.get("optional")),
        "vocabulary": vocabulary_scope(str(requirement["path"])),
        "message": message,
    }
    if "value" in requirement:
        diagnostic["expected"] = requirement["value"]
    return diagnostic


def pair_diagnostic(
    *,
    code: str,
    path: str,
    expected: Any,
    actual: Any,
    message: str,
    severity: str = "error",
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "path": path,
        "expected": expected,
        "actual": actual,
        "optional": severity == "warning",
        "vocabulary": vocabulary_scope(path),
        "message": message,
    }


__all__ = [
    "CORE_CAPABILITIES",
    "pair_diagnostic",
    "requirement_diagnostic",
    "vocabulary_scope",
]
