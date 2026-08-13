from __future__ import annotations

import re
from pathlib import Path


_OBJECT_ID = re.compile(r"[0-9a-fA-F]{40,64}")
_SMALL_METADATA_LIMIT = 8192
_PACKED_REFS_LIMIT = 16 * 1024 * 1024
_INVALID_REF_CHARS = frozenset(" ~^:?*[\\")


def normalize_object_id(text: object) -> str | None:
    """Return a canonical Git object id, or ``None`` for non-object text."""
    if not isinstance(text, str):
        return None
    value = text.strip()
    return value.lower() if _OBJECT_ID.fullmatch(value) else None


def _read_text(path: Path, *, limit: int = _SMALL_METADATA_LIMIT) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > limit:
            return None
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            return None
        return raw.decode("utf-8", errors="strict")
    except (OSError, UnicodeError, ValueError):
        return None


def _git_directory(root: Path) -> Path | None:
    marker = root / ".git"
    try:
        if marker.is_dir():
            return marker.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    text = _read_text(marker)
    if text is None:
        return None
    line = text.strip()
    prefix = "gitdir:"
    if not line.lower().startswith(prefix):
        return None
    raw = Path(line[len(prefix):].strip()).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    try:
        candidate = candidate.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        return candidate if candidate.is_dir() else None
    except (OSError, ValueError):
        return None


def _common_directories(git_dir: Path) -> list[Path]:
    out = [git_dir]
    marker = git_dir / "commondir"
    text = _read_text(marker)
    if text is not None:
        try:
            raw = Path(text.strip())
            common = (raw if raw.is_absolute() else git_dir / raw).resolve()
            if common.is_dir() and common not in out:
                out.append(common)
        except (OSError, RuntimeError, ValueError):
            pass
    return out


def _valid_ref(value: str) -> str | None:
    ref = value.strip()
    parts = ref.split("/")
    if (
        not ref
        or len(ref) > 1024
        or not ref.startswith("refs/")
        or ref.endswith(("/", "."))
        or "//" in ref
        or ".." in ref
        or "@{" in ref
        or any(ord(char) < 32 or ord(char) == 127 or char in _INVALID_REF_CHARS for char in ref)
        or any(not part or part.startswith(".") or part.endswith((".", ".lock")) for part in parts)
    ):
        return None
    return ref


def _packed_ref(git_dir: Path, ref: str) -> str | None:
    for base in _common_directories(git_dir):
        path = base / "packed-refs"
        try:
            if not path.is_file() or path.stat().st_size > _PACKED_REFS_LIMIT:
                continue
            rows = path.open("r", encoding="utf-8", errors="strict")
        except (OSError, UnicodeError, ValueError):
            continue
        try:
            with rows:
                for row in rows:
                    if not row or row.startswith(("#", "^")):
                        continue
                    fields = row.rstrip("\r\n").split(" ", 1)
                    if len(fields) == 2 and fields[1].strip() == ref:
                        object_id = normalize_object_id(fields[0])
                        if object_id:
                            return object_id
        except (OSError, UnicodeError, ValueError):
            continue
    return None


def read_git_head(root: str | Path) -> str | None:
    """Read the checked-out commit without requiring the ``git`` executable.

    This is a narrow fallback for minimal runtime containers.  It identifies the
    checkout from Git's own HEAD/ref metadata; it does not establish cleanliness.
    Malformed paths, symbolic-ref loops and non-object values fail closed.
    """
    try:
        repository = Path(root).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    git_dir = _git_directory(repository)
    if git_dir is None:
        return None
    head = _read_text(git_dir / "HEAD")
    if head is None:
        return None
    value = head.strip()
    for _ in range(8):
        object_id = normalize_object_id(value)
        if object_id:
            return object_id
        if not value.startswith("ref:"):
            return None
        ref = _valid_ref(value[4:])
        if ref is None:
            return None
        next_value = None
        for base in _common_directories(git_dir):
            path = base.joinpath(*ref.split("/"))
            text = _read_text(path)
            if text is not None:
                next_value = text.strip()
            if next_value:
                break
        if next_value is None:
            return _packed_ref(git_dir, ref)
        value = next_value
    return None
