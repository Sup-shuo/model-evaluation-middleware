from __future__ import annotations

import json

from model_evaluation.core.security import redact_diagnostic


def dump(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def doctor_dump(obj, orchestrator) -> None:
    """Backward-compatible JSON renderer used by integrations and tests."""

    dump(redact_diagnostic(obj, orchestrator.pm.secrets.redaction_values()))
