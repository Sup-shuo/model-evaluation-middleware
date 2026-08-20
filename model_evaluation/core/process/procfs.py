from __future__ import annotations

import os
from pathlib import Path


def proc_start_ticks(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2 :].split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


def proc_pgid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (OSError, ProcessLookupError):
        return None


def proc_sid(pid: int) -> int | None:
    try:
        return os.getsid(pid)
    except (OSError, ProcessLookupError):
        return None


def linux_boot_id() -> str | None:
    """Return the current Linux boot identity used to scope ownership records."""
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


def proc_group_snapshot(pgid: int | None) -> tuple[dict[int, int], dict[int, int], bool]:
    """Return live members, zombies and snapshot completeness for one group."""
    if pgid is None:
        return {}, {}, True
    live: dict[int, int] = {}
    zombies: dict[int, int] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return live, zombies, False
    complete = True
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            tail = raw[raw.rfind(")") + 2 :].split()
            if int(tail[2]) != int(pgid):
                continue
            target = zombies if tail[0] == "Z" else live
            target[int(entry.name)] = int(tail[19])
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError):
            complete = False
    return live, zombies, complete


def proc_group_members(pgid: int | None) -> dict[int, int]:
    """Return ``{pid: start_ticks}`` for live observable group members."""
    live, _zombies, _complete = proc_group_snapshot(pgid)
    return live
