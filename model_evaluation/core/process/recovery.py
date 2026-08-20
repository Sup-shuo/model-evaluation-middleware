from __future__ import annotations

import os
import signal
import time

from model_evaluation.core.process.procfs import (
    linux_boot_id,
    proc_pgid,
    proc_start_ticks,
)
from model_evaluation.core.serialization import json_loads_strict


def _invalid(path, *, error: str, pid: int | None = None, pgid: int | None = None) -> dict:
    row = {"path": str(path), "status": "invalid", "error": error}
    if pid is not None:
        row["pid"] = pid
    if pgid is not None:
        row["pgid"] = pgid
    return row


def recover_stale_managed(
    manager,
    *,
    grace_seconds: float,
    kill_seconds: float,
    boot_id_fn=linux_boot_id,
    start_ticks_fn=proc_start_ticks,
    pgid_fn=proc_pgid,
) -> list[dict]:
    """Recover stale owned processes without signalling ambiguous identities."""
    if not manager.ownership_root:
        return []
    results: list[dict] = []
    for path in sorted(manager.ownership_root.glob("process-*.json")):
        if path.is_symlink():
            results.append(_invalid(path, error="ownership record is a symlink"))
            continue
        try:
            record = json_loads_strict(path.read_text(encoding="utf-8"))
            pid = int(record["pid"])
            pgid = int(record.get("pgid") or pid)
            expected_ticks = record.get("start_ticks")
        except Exception as exc:
            results.append(_invalid(path, error=str(exc)))
            continue

        recorded_boot = record.get("boot_id")
        current_boot = boot_id_fn()
        if current_boot and not recorded_boot:
            results.append(
                _invalid(
                    path,
                    error="ownership record is missing boot_id",
                    pid=pid,
                    pgid=pgid,
                )
            )
            continue
        if recorded_boot and current_boot and recorded_boot != current_boot:
            path.unlink(missing_ok=True)
            results.append(
                {"path": str(path), "pid": pid, "pgid": pgid, "status": "expired_boot"}
            )
            continue

        current_ticks = start_ticks_fn(pid)
        if current_ticks is None:
            if manager._group_alive(pgid):
                results.append(
                    {
                        "path": str(path),
                        "pid": pid,
                        "pgid": pgid,
                        "status": "orphaned_group_ambiguous",
                    }
                )
                continue
            path.unlink(missing_ok=True)
            results.append({"path": str(path), "pid": pid, "status": "gone"})
            continue

        current_pgid = pgid_fn(pid)
        identity_matches = (
            expected_ticks is not None
            and current_ticks == expected_ticks
            and current_pgid is not None
            and pgid == current_pgid
            and pgid == pid
        )
        if not identity_matches:
            results.append(
                {
                    "path": str(path),
                    "pid": pid,
                    "status": "identity_mismatch",
                    "recorded_pgid": pgid,
                    "current_pgid": current_pgid,
                }
            )
            continue

        try:
            os.killpg(pgid, signal.SIGTERM)
            _wait_group(manager, pgid, grace_seconds)
            if manager._group_alive(pgid):
                os.killpg(pgid, signal.SIGKILL)
                _wait_group(manager, pgid, kill_seconds)
            if manager._group_alive(pgid):
                results.append(
                    {"path": str(path), "pid": pid, "status": "cleanup_failed"}
                )
                continue
        except ProcessLookupError:
            pass
        path.unlink(missing_ok=True)
        results.append({"path": str(path), "pid": pid, "status": "recovered"})
    return results


def _wait_group(manager, pgid: int, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and manager._group_alive(pgid):
        time.sleep(0.05)
