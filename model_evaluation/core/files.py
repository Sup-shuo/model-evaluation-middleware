from __future__ import annotations

import os
import tempfile
from pathlib import Path

from model_evaluation.core.serialization import json_dumps_strict


def atomic_json(path: str | Path, obj: object) -> None:
    """Atomically replace a JSON file without exposing a partial document."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f"{target.name}.tmp.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json_dumps_strict(obj, ensure_ascii=False, indent=2, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_text(path: str | Path, text: str) -> None:
    """Atomically replace a UTF-8 text file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f"{target.name}.tmp.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
