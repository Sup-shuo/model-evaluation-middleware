from __future__ import annotations

import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from model_evaluation.core.errors import ModelEvalError, ProcessError
from model_evaluation.core.identifiers import stable_id
from model_evaluation.core.process.env import apply_env_patch
from model_evaluation.core.process.procfs import (
    linux_boot_id as _linux_boot_id,
    proc_group_members as _proc_group_members,
    proc_group_snapshot as _proc_group_snapshot,
    proc_pgid as _proc_pgid,
    proc_sid as _proc_sid,
    proc_start_ticks as _proc_start_ticks,
)
from model_evaluation.core.process.recovery import recover_stale_managed
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.core.security import execution_subprocess_env
from model_evaluation.core.serialization import json_dumps_strict, json_loads_strict

@dataclass
class ProcessHandle:
    process: subprocess.Popen
    spec: dict
    stdout_handle: IO | None = None
    stderr_handle: IO | None = None
    ownership_path: Path | None = None
    start_ticks: int | None = None
    pgid: int | None = None

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self):
        return self.process.poll()


class SecretStore:
    def __init__(self, values: dict[str, str] | None = None):
        self.values = values or {}
        self._resolved_values: set[str] = {str(v) for v in self.values.values() if v}

    def resolve(self, ref: str) -> str:
        if ref in self.values:
            value = self.values[ref]
            if value:
                self._resolved_values.add(str(value))
            return value
        if ref.startswith("secret://env/"):
            name = ref.removeprefix("secret://env/")
            if name in os.environ:
                value = os.environ[name]
                if value:
                    self._resolved_values.add(str(value))
                return value
        raise ProcessError(f"secret reference unresolved: {ref}")

    def redaction_values(self) -> tuple[str, ...]:
        return tuple(sorted(self._resolved_values, key=len, reverse=True))


class ProcessManager:
    def __init__(
        self,
        schemas: SchemaStore,
        *,
        secrets: SecretStore | None = None,
        ownership_root: str | Path | None = None,
    ):
        self.schemas = schemas
        self.secrets = secrets or SecretStore()
        self.ownership_root = Path(ownership_root).resolve() if ownership_root else None
        if self.ownership_root:
            self.ownership_root.mkdir(parents=True, exist_ok=True)

    def _stdio(self, config: dict | None, *, stream: str):
        config = config or {"mode": "inherit"}
        mode = config.get("mode", "inherit")
        if mode == "inherit":
            return None, None
        if mode == "capture":
            return subprocess.PIPE, None
        if mode == "merge_stdout" and stream == "stderr":
            return subprocess.STDOUT, None
        if mode == "file":
            path = Path(config["path"]).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("ab")
            return handle, handle
        raise ProcessError(f"unsupported {stream} mode: {mode}")

    def prepare_env(self, spec: dict, base_env: dict[str, str] | None = None) -> dict[str, str]:
        base = execution_subprocess_env() if base_env is None else dict(base_env)
        env = apply_env_patch(base, spec.get("env_patch"))
        for key, ref in (spec.get("secret_env") or {}).items():
            env[key] = self.secrets.resolve(ref)
        return env

    def _ownership_record(self, handle: ProcessHandle) -> dict:
        return {
            "schema_version": "1.0",
            "pid": handle.pid,
            "pgid": handle.pgid,
            "start_ticks": handle.start_ticks,
            "boot_id": _linux_boot_id(),
            "argv0": str(handle.spec["argv"][0]),
            "spec_id": stable_id(handle.spec),
            "metadata": handle.spec.get("metadata") or {},
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "state": "running",
        }

    def _write_ownership(self, handle: ProcessHandle) -> None:
        if not self.ownership_root:
            return
        path = self.ownership_root / f"process-{handle.pid}-{uuid.uuid4().hex[:12]}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json_dumps_strict(self._ownership_record(handle), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        handle.ownership_path = path


    def _abort_uncommitted(self, handle: ProcessHandle) -> dict:
        """Best-effort abort for a process that never completed ownership commit."""
        # If identity is complete, use the normal ownership-safe group cleanup.
        if handle.start_ticks is not None and handle.pgid is not None:
            return self.stop_with_report(handle, grace_seconds=1.0, kill_seconds=1.0)

        report = {
            "status": "clean",
            "pid": handle.pid,
            "pgid": handle.pgid,
            "graceful": {"attempted": True, "target": "leader", "result": "unknown"},
            "fallback": {"sigkill_attempted": False, "target": "leader", "result": "not_needed"},
            "owned_process_group_remaining": False,
        }
        try:
            self._direct_stop(handle.process, grace_seconds=1.0, kill_seconds=1.0)
            report["graceful"]["result"] = "success"
        except BaseException as exc:
            report["status"] = "incomplete"
            report["graceful"]["result"] = "error"
            self._record_cleanup_error(report, "abort_uncommitted_leader", exc)

        if handle.pgid is None:
            # Without a group identity Core cannot prove descendants are gone.
            report["status"] = "incomplete"
            report["owned_process_group_remaining"] = True
        elif self._group_alive(handle.pgid):
            report["status"] = "incomplete"
            report["owned_process_group_remaining"] = True

        if report["status"] == "clean":
            return self._finalize_clean_handle(handle, report)
        return report

    @staticmethod
    def _remove_ownership(handle: ProcessHandle) -> None:
        if handle.ownership_path:
            handle.ownership_path.unlink(missing_ok=True)

    def start(self, spec: dict, *, base_env: dict[str, str] | None = None) -> ProcessHandle:
        self.schemas.validate("process_spec", spec)
        for stream in ("stdin", "stdout", "stderr"):
            cfg = spec.get(stream) or {}
            if cfg.get("mode") == "file" and not cfg.get("path"):
                raise ProcessError(f"ProcessSpec {stream}.mode=file requires path")

        stdout_handle = None
        stderr_handle = None
        stdin_handle = None
        proc = None
        try:
            stdout_target, stdout_handle = self._stdio(spec.get("stdout"), stream="stdout")
            stderr_target, stderr_handle = self._stdio(spec.get("stderr"), stream="stderr")
            stdin_cfg = spec.get("stdin") or {"mode": "null"}
            stdin_mode = stdin_cfg.get("mode", "null")
            stdin_target = None if stdin_mode == "inherit" else subprocess.DEVNULL
            if stdin_mode == "file":
                stdin_handle = Path(stdin_cfg["path"]).open("rb")
                stdin_target = stdin_handle
            proc = subprocess.Popen(
                spec["argv"], cwd=spec.get("cwd") or None, env=self.prepare_env(spec, base_env),
                stdin=stdin_target, stdout=stdout_target, stderr=stderr_target,
                start_new_session=True, close_fds=True,
            )
        except BaseException as exc:
            for handle in (stdin_handle, stdout_handle, stderr_handle):
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass
            if isinstance(exc, (KeyboardInterrupt, SystemExit, ModelEvalError)):
                raise
            raise ProcessError(f"failed to prepare/start process: {exc}") from exc

        # Popen has returned: from this point onward Core always owns a
        # provisional handle.  No identity/commit failure may lose the child.
        handle = ProcessHandle(
            proc, spec, stdout_handle, stderr_handle,
            start_ticks=None, pgid=proc.pid,
        )
        commit_error: BaseException | None = None
        try:
            if stdin_handle is not None:
                stdin_handle.close()
                stdin_handle = None
            handle.start_ticks = _proc_start_ticks(proc.pid)
            observed_pgid = _proc_pgid(proc.pid)
            if observed_pgid is not None:
                handle.pgid = observed_pgid
            if handle.start_ticks is None or observed_pgid is None:
                raise ProcessError(
                    f"failed to establish process ownership identity: pid={handle.pid} "
                    f"start_ticks={handle.start_ticks!r} pgid={observed_pgid!r}"
                )
            self._write_ownership(handle)
            return handle
        except BaseException as exc:
            commit_error = exc

        report = self._abort_uncommitted(handle)
        cleanup_status = report.get("status")
        if isinstance(commit_error, ModelEvalError):
            commit_error.details.setdefault("cleanup_status", cleanup_status)
            commit_error.details.setdefault("cleanup_report", report)
            if cleanup_status == "incomplete":
                try:
                    setattr(commit_error, "_model_eval_cleanup_status", "incomplete")
                except Exception:
                    pass
            raise commit_error
        if isinstance(commit_error, (KeyboardInterrupt, SystemExit)):
            if cleanup_status == "incomplete":
                try:
                    setattr(commit_error, "_model_eval_cleanup_status", "incomplete")
                except Exception:
                    pass
            raise commit_error

        err = ProcessError(
            f"process start was not committed: {type(commit_error).__name__}: {commit_error}",
            details={"cleanup_status": cleanup_status, "cleanup_report": report},
        )
        if cleanup_status == "incomplete":
            try:
                setattr(err, "_model_eval_cleanup_status", "incomplete")
            except Exception:
                pass
        raise err from commit_error

    @staticmethod
    def _attach_cleanup_error(primary: BaseException, cleanup_exc: BaseException) -> None:
        """Attach cleanup diagnostics without replacing the primary failure."""
        try:
            setattr(primary, "_model_eval_cleanup_error", cleanup_exc)
        except Exception:
            pass
        cleanup_status = None
        details = getattr(cleanup_exc, "details", None)
        if isinstance(details, dict):
            cleanup_status = details.get("cleanup_status")
        cleanup_status = cleanup_status or getattr(cleanup_exc, "_model_eval_cleanup_status", None)
        if cleanup_status == "incomplete":
            try:
                setattr(primary, "_model_eval_cleanup_status", "incomplete")
            except Exception:
                pass
        if isinstance(primary, ModelEvalError):
            primary.details.setdefault("cleanup_error", {
                "type": type(cleanup_exc).__name__,
                "message": str(cleanup_exc),
            })
            if cleanup_status == "incomplete":
                primary.details["cleanup_status"] = "incomplete"
        elif hasattr(primary, "add_note"):
            try:
                primary.add_note(
                    f"secondary process cleanup failure: {type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            except Exception:
                pass

    def run(self, spec: dict, *, base_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        self.schemas.validate("process_spec", spec)
        timeout = spec.get("timeout_seconds")
        handle = self.start(spec, base_env=base_env)
        try:
            out, err = handle.process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            primary = ProcessError(f"process timed out after {timeout}s: {spec['argv'][0]}")
            try:
                self.stop(handle)
            except BaseException as cleanup_exc:
                self._attach_cleanup_error(primary, cleanup_exc)
            raise primary from exc
        except BaseException as exc:
            try:
                self.stop(handle)
            except BaseException as cleanup_exc:
                # Cleanup is secondary information.  Never replace the original
                # failure/interrupt with a cleanup exception.
                self._attach_cleanup_error(exc, cleanup_exc)
            raise
        else:
            # The owned unit is the entire process group, not only its leader.
            # A wrapper/evaluator that returns while descendants remain would
            # otherwise leak workers into the next run.  Give the kernel and
            # short-lived framework helpers a small bounded settling window:
            # killpg(0) and the following /proc scan are not atomic, so a group
            # can disappear between them and look conservatively "alive" for
            # one observation even though no member remains.
            if not self._wait_owned_group_gone(handle, handle.pgid, 0.5):
                primary = ProcessError(
                    f"process leader exited while owned process group remained alive: "
                    f"pid={handle.pid} pgid={handle.pgid}"
                )
                try:
                    self.stop(handle)
                except BaseException as cleanup_exc:
                    self._attach_cleanup_error(primary, cleanup_exc)
                raise primary
            # The workload has already completed.  Housekeeping must not turn a
            # completed workload into a new primary failure.
            try:
                self._remove_ownership(handle)
            except BaseException:
                pass
            try:
                self._close(handle)
            except BaseException:
                pass
            return subprocess.CompletedProcess(spec["argv"], handle.process.returncode, out, err)

    @staticmethod
    def _group_alive(pgid: int | None) -> bool:
        if pgid is None:
            return False
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # A group that exists but cannot be signalled is still alive.
            return True
        live, zombies, complete = _proc_group_snapshot(pgid)
        if live:
            return True
        if zombies and complete:
            # Docker/container PID 1 may leave killed descendants as zombies.
            # They are terminated even though the numeric PGID still exists.
            return False
        # No observable member is ambiguous (for example non-Linux procfs or a
        # concurrent procfs race), so retain the conservative kernel answer.
        return True

    @staticmethod
    def _direct_stop(proc: subprocess.Popen, *, grace_seconds: float, kill_seconds: float) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            proc.kill()
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=kill_seconds)
        except subprocess.TimeoutExpired as exc:
            raise ProcessError(
                f"process did not terminate: pid={proc.pid}"
            ) from exc

    @staticmethod
    def _signal_number(name: str) -> int:
        mapping = {"SIGTERM": signal.SIGTERM, "SIGINT": signal.SIGINT}
        if name not in mapping:
            raise ProcessError(f"unsupported graceful shutdown signal: {name}")
        return mapping[name]

    @staticmethod
    def _wait_owned_group_gone(handle: ProcessHandle, pgid: int | None, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while True:
            handle.process.poll()
            leader_gone = handle.process.poll() is not None
            group_gone = not ProcessManager._group_alive(pgid)
            if leader_gone and group_gone:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    @staticmethod
    def _owned_group_status(handle: ProcessHandle, pgid: int | None, known_members: dict[int, int] | None = None) -> str:
        """Return gone/verified/ambiguous without trusting a bare PGID.

        While the original leader is alive, PID + start ticks + PGID prove the
        group identity.  If the leader exits during graceful shutdown, a
        previously observed group member with matching start ticks can continue
        that proof.  Otherwise a surviving PGID is treated as ambiguous and is
        never signalled blindly on a shared server.
        """
        if not ProcessManager._group_alive(pgid):
            return "gone"
        current_start = _proc_start_ticks(handle.pid)
        if current_start is not None:
            current_pgid = _proc_pgid(handle.pid)
            if (
                handle.start_ticks is not None
                and current_start == handle.start_ticks
                and current_pgid == pgid
                and pgid == handle.pid
            ):
                return "verified"
            return "ambiguous"
        current_members = _proc_group_members(pgid)
        if known_members and any(current_members.get(pid) == ticks for pid, ticks in known_members.items()):
            return "verified"
        # start_new_session=True gives every descendant the original leader PID
        # as its session ID.  If that leader PID is currently absent, a living
        # member still in that session is strong evidence that this is the
        # original orphaned group, not a numerically reused PGID.
        if pgid == handle.pid and any(_proc_sid(pid) == handle.pid for pid in current_members):
            return "verified"
        return "ambiguous"

    @staticmethod
    def _record_cleanup_error(report: dict, phase: str, exc: BaseException) -> None:
        report.setdefault("secondary_errors", []).append({
            "phase": phase,
            "type": type(exc).__name__,
            "message": str(exc),
        })

    def _finalize_clean_handle(self, handle: ProcessHandle, report: dict) -> dict:
        """Best-effort housekeeping after process ownership is already clean."""
        try:
            self._remove_ownership(handle)
        except BaseException as exc:
            self._record_cleanup_error(report, "remove_ownership", exc)
        try:
            self._close(handle)
        except BaseException as exc:
            self._record_cleanup_error(report, "close_handles", exc)
        return report

    def stop_with_report(
        self,
        handle: ProcessHandle,
        *,
        graceful_signal: str = "SIGTERM",
        grace_seconds: float = 10.0,
        kill_seconds: float = 3.0,
    ) -> dict:
        """Bounded, idempotent cleanup for one Core-owned process group.

        Graceful shutdown is leader-first so the inference framework can run
        its own worker/communicator/runtime teardown.  Only after that bounded
        opportunity may Core use a verified owned process group as a SIGKILL
        fallback.  A bare/reused PGID is never trusted, and cleanup never
        inspects accelerator memory or unrelated device processes.

        Cleanup-internal errors are contained in the returned report; this
        method is intentionally safe to call from an exception handler.
        """
        pgid = handle.pgid or _proc_pgid(handle.pid)
        report = {
            "status": "clean",
            "pid": handle.pid,
            "pgid": pgid,
            "ownership": {"initial": "unknown", "fallback": "not_needed"},
            "graceful": {
                "attempted": False,
                "target": "leader",
                "signal": graceful_signal,
                "timeout_seconds": float(grace_seconds),
                "result": "not_needed",
            },
            "fallback": {
                "sigkill_attempted": False,
                "target": "process_group",
                "timeout_seconds": float(kill_seconds),
                "result": "not_needed",
            },
            "owned_process_group_remaining": False,
        }

        try:
            initial_status = self._owned_group_status(handle, pgid)
        except BaseException as exc:
            self._record_cleanup_error(report, "initial_ownership_check", exc)
            initial_status = "ambiguous"
        report["ownership"]["initial"] = initial_status

        if initial_status == "gone":
            report["graceful"]["result"] = "already_exited"
            return self._finalize_clean_handle(handle, report)
        if initial_status != "verified":
            report["status"] = "incomplete"
            report["graceful"]["result"] = "ownership_ambiguous"
            report["owned_process_group_remaining"] = True
            return report

        # Capture concrete member identities while the leader still proves the
        # group.  These allow a safe fallback if the leader exits before its
        # workers do.
        try:
            known_members = _proc_group_members(pgid)
        except BaseException as exc:
            self._record_cleanup_error(report, "capture_group_members", exc)
            known_members = {}
        if handle.start_ticks is not None:
            known_members.setdefault(handle.pid, handle.start_ticks)

        try:
            sig = self._signal_number(graceful_signal)
        except BaseException as exc:
            report["graceful"]["result"] = "error"
            report["graceful"]["error"] = f"{type(exc).__name__}: {exc}"
            self._record_cleanup_error(report, "resolve_graceful_signal", exc)
            sig = signal.SIGTERM

        report["graceful"]["attempted"] = True
        try:
            # Leader-first: let the framework coordinate its own shutdown.
            if handle.process.poll() is None:
                handle.process.send_signal(sig)
            else:
                report["graceful"]["result"] = "already_exited"
        except ProcessLookupError:
            report["graceful"]["result"] = "already_exited"
        except BaseException as exc:
            report["graceful"]["result"] = "error"
            report["graceful"]["error"] = f"{type(exc).__name__}: {exc}"
            self._record_cleanup_error(report, "graceful_signal", exc)

        try:
            graceful_gone = self._wait_owned_group_gone(handle, pgid, grace_seconds)
        except BaseException as exc:
            self._record_cleanup_error(report, "graceful_wait", exc)
            graceful_gone = False
        if graceful_gone:
            if report["graceful"]["result"] not in {"already_exited", "error"}:
                report["graceful"]["result"] = "success"
            return self._finalize_clean_handle(handle, report)

        if report["graceful"]["result"] not in {"error", "already_exited"}:
            report["graceful"]["result"] = "timeout"

        try:
            fallback_ownership = self._owned_group_status(handle, pgid, known_members)
        except BaseException as exc:
            self._record_cleanup_error(report, "fallback_ownership_check", exc)
            fallback_ownership = "ambiguous"
        report["ownership"]["fallback"] = fallback_ownership
        if fallback_ownership == "gone":
            return self._finalize_clean_handle(handle, report)
        if fallback_ownership != "verified":
            report["status"] = "incomplete"
            report["fallback"]["result"] = "ownership_ambiguous"
            report["owned_process_group_remaining"] = True
            return report

        # Framework shutdown did not complete.  Fall back only to a process
        # group whose ownership is still evidenced by PID/start-ticks.
        report["fallback"]["sigkill_attempted"] = True
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            elif handle.process.poll() is None:
                handle.process.kill()
        except ProcessLookupError:
            pass
        except BaseException as exc:
            report["fallback"]["result"] = "error"
            report["fallback"]["error"] = f"{type(exc).__name__}: {exc}"
            self._record_cleanup_error(report, "fallback_sigkill", exc)

        try:
            fallback_gone = self._wait_owned_group_gone(handle, pgid, kill_seconds)
        except BaseException as exc:
            self._record_cleanup_error(report, "fallback_wait", exc)
            fallback_gone = False
        if fallback_gone:
            if report["fallback"]["result"] != "error":
                report["fallback"]["result"] = "success"
            return self._finalize_clean_handle(handle, report)

        report["status"] = "incomplete"
        report["owned_process_group_remaining"] = True
        try:
            final_ownership = self._owned_group_status(handle, pgid, known_members)
        except BaseException as exc:
            self._record_cleanup_error(report, "final_ownership_check", exc)
            final_ownership = "ambiguous"
        report["ownership"]["final"] = final_ownership
        if report["fallback"]["result"] != "error":
            report["fallback"]["result"] = "ownership_ambiguous" if final_ownership == "ambiguous" else "timeout"
        # Keep the ownership record for conservative stale recovery.  Do not
        # close output handles while an owned/ambiguous group may still write.
        return report

    def stop(
        self,
        handle: ProcessHandle,
        *,
        grace_seconds: float = 10.0,
        kill_seconds: float = 3.0,
        graceful_signal: str = "SIGTERM",
    ) -> None:
        report = self.stop_with_report(
            handle,
            graceful_signal=graceful_signal,
            grace_seconds=grace_seconds,
            kill_seconds=kill_seconds,
        )
        if report["status"] != "clean":
            err = ProcessError(
                f"process group did not terminate: pid={handle.pid} pgid={report.get('pgid')}",
                details={"cleanup_status": "incomplete", "cleanup_report": report},
            )
            try:
                setattr(err, "_model_eval_cleanup_status", "incomplete")
            except Exception:
                pass
            raise err

    def recover_stale_managed(self, *, grace_seconds: float = 1.0, kill_seconds: float = 1.0) -> list[dict]:
        return recover_stale_managed(
            self,
            grace_seconds=grace_seconds,
            kill_seconds=kill_seconds,
            boot_id_fn=_linux_boot_id,
            start_ticks_fn=_proc_start_ticks,
            pgid_fn=_proc_pgid,
        )

    def _close(self, handle: ProcessHandle) -> None:
        for h in (handle.stdout_handle, handle.stderr_handle, handle.process.stdout, handle.process.stderr):
            if h is not None and hasattr(h, "closed") and not h.closed:
                h.close()
