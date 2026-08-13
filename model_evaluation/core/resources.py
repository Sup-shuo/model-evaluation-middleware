from __future__ import annotations

import fcntl
import hashlib
import socket
from contextlib import ExitStack, contextmanager
from pathlib import Path

from model_evaluation.core.errors import ResourceError


def _key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


class FileLease:
    def __init__(self, path: Path, *, exclusive: bool = True):
        self.path = path
        self.exclusive = exclusive
        self.handle = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        op = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(self.handle.fileno(), op | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close(); self.handle = None
            raise ResourceError(f"resource is already claimed: {self.path.name}") from exc

    def release(self):
        if self.handle:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close(); self.handle = None

    def __enter__(self): self.acquire(); return self
    def __exit__(self, *_): self.release()


class ResourceManager:
    def __init__(self, runtime_root: str | Path):
        self.runtime_root = Path(runtime_root).resolve()
        self.locks_root = self.runtime_root / "claims"

    def _lease_for(self, claim: dict) -> FileLease:
        kind = str(claim["kind"]); ident = str(claim["id"])
        return FileLease(self.locks_root / kind / f"{_key(ident)}.lock", exclusive=bool(claim.get("exclusive", True)))

    @staticmethod
    def check_port(host: str, port: int) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Managed HTTP servers normally enable SO_REUSEADDR.  Mirror
                # that bind behavior here so a just-stopped server's TCP
                # TIME_WAIT sockets do not make the next serial Matrix item
                # look like an active listener.  A currently bound listener
                # still prevents this bind.
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                s.listen(1)
            except OSError as exc:
                raise ResourceError(f"port unavailable: {host}:{port}") from exc

    @contextmanager
    def acquire(self, claims: list[dict]):
        """Acquire all framework-visible resource claims in deterministic order.

        Port claims use a framework lock plus an OS availability check.  The
        socket itself cannot remain bound while the backend starts, so the lock
        prevents races among this framework's runs while `check_port` detects
        external users immediately before service start.
        """
        ordered = sorted(claims, key=lambda c: (str(c["kind"]), str(c["id"])))
        with ExitStack() as stack:
            for claim in ordered:
                stack.enter_context(self._lease_for(claim))
                if claim["kind"] == "port":
                    self.check_port(str(claim.get("host") or "127.0.0.1"), int(claim["id"]))
            yield

    @contextmanager
    def run_lock(self):
        # Backward-compatible convenience; new Orchestrator uses acquire(plan.resources).
        claim = {"kind": "run_lock", "id": "global-orchestrator", "exclusive": True}
        with self.acquire([claim]):
            yield
