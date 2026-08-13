from __future__ import annotations

import os
import sys
import subprocess
import uuid
import time
from dataclasses import dataclass
from pathlib import Path

from model_evaluation.core.errors import AdapterExecutionError, AdapterProtocolError, ConfigError
from model_evaluation.core.registry.operation_contracts import validate_operation_input, validate_operation_output
from model_evaluation.core.schema.validator import SchemaStore
from model_evaluation.core.security import adapter_subprocess_env, redact_text
from model_evaluation.core.serialization import json_dumps_strict, json_loads_strict

PROTOCOL_VERSION = "1.0"
VALID_KINDS = {"device", "runtime", "environment", "backend", "dataset", "binding", "evaluator"}
SUPPORTED_SHARED_SCHEMAS = {
    "device_descriptor":"1.0", "runtime_descriptor":"1.0", "environment_descriptor":"1.0",
    "process_spec":"1.0", "service_descriptor":"1.0", "dataset_artifact":"1.0",
    "framework_task_artifact":"1.0", "canonical_result":"1.0", "requirement_set":"1.0",
    "capability_set":"1.0", "env_patch":"1.0", "backend_start_plan":"1.1",
    "backend_preflight_plan":"1.0", "preflight_probe_result":"1.0",
}

def _validate_adapter_name(name: str) -> None:
    if not name or "/" in name or "\\" in name or name in {".",".."}:
        raise AdapterProtocolError(f"adapter name must be one filesystem component: {name!r}")

def _validate_schema_versions(manifest: dict) -> None:
    for schema_name, version in (manifest.get("schema_versions") or {}).items():
        supported=SUPPORTED_SHARED_SCHEMAS.get(schema_name)
        if supported is None:
            raise AdapterProtocolError(f"adapter declares unknown shared schema {schema_name!r}")
        if version != supported:
            raise AdapterProtocolError(f"adapter shared schema {schema_name} version {version!r} is incompatible; Core supports {supported}")

@dataclass(frozen=True)
class AdapterIdentity:
    kind: str
    name: str
    version: str
    path: Path
    manifest: dict

class AdapterClient:
    def __init__(self, identity: AdapterIdentity, schemas: SchemaStore, *, default_timeout: float = 5.0):
        self.identity = identity
        self.schemas = schemas
        self.default_timeout = default_timeout
        self.last_warnings: list[str] = []

    def invoke(self, operation: str, input_obj: dict, *, context: dict | None = None, timeout: float | None = None) -> dict:
        if operation not in self.identity.manifest.get("operations", []):
            raise AdapterProtocolError(f"adapter {self.identity.kind}/{self.identity.name} does not support {operation}")
        validate_operation_input(self.schemas, self.identity.kind, operation, input_obj)
        request = {
            "api_version": PROTOCOL_VERSION,
            "request_id": f"req-{uuid.uuid4().hex[:16]}",
            "operation": operation,
            "input": input_obj,
        }
        request_context = dict(context or {})
        # Generic controller identity.  Environment providers may use this when
        # "current" means the interpreter running Core rather than the adapter
        # subprocess interpreter resolved from PATH.  Callers cannot override it.
        request_context["controller_python"] = sys.executable
        request["context"] = request_context
        self.schemas.validate("adapter_request", request)
        try:
            proc = subprocess.run(
                [str(self.identity.path), "invoke"],
                input=json_dumps_strict(request, ensure_ascii=False),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout or self.default_timeout,
                check=False,
                env=adapter_subprocess_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise AdapterProtocolError(
                f"adapter {self.identity.kind}/{self.identity.name} operation {operation} timed out"
            ) from exc
        try:
            response = json_loads_strict(proc.stdout)
        except Exception as exc:
            secret_values=[str(input_obj.get('auth_value'))] if input_obj.get('auth_value') else []
            stderr = redact_text(proc.stderr.strip()[-2000:],secret_values)
            raise AdapterProtocolError(
                f"adapter {self.identity.kind}/{self.identity.name} returned invalid JSON; stderr={stderr!r}"
            ) from exc
        self.schemas.validate("adapter_response", response)
        if response.get("request_id") != request["request_id"]:
            raise AdapterProtocolError("adapter response request_id mismatch")
        if response.get("api_version") != PROTOCOL_VERSION:
            raise AdapterProtocolError(f"adapter response API version must be exactly {PROTOCOL_VERSION}")
        self.last_warnings = [str(x) for x in (response.get("warnings") or [])]
        # Protocol v1: ok=true must exit 0; ok=false must exit non-zero.
        if response["ok"] and proc.returncode != 0:
            raise AdapterProtocolError(
                f"adapter {self.identity.kind}/{self.identity.name} returned ok=true with rc={proc.returncode}"
            )
        if not response["ok"] and proc.returncode == 0:
            raise AdapterProtocolError(
                f"adapter {self.identity.kind}/{self.identity.name} returned ok=false with rc=0"
            )
        if not response["ok"]:
            err = response["error"]
            raise AdapterExecutionError(
                err["code"], redact_text(err["message"],[str(input_obj.get('auth_value'))] if input_obj.get('auth_value') else []), retryable=bool(err.get("retryable")), details=err.get("details") or {}
            )
        output = response["output"]
        validate_operation_output(self.schemas, self.identity.kind, operation, output, input_obj=input_obj)
        return output

class AdapterRegistry:
    def __init__(self, root: str | Path, schemas: SchemaStore, *, manifest_timeout: float = 1.0, discovery_timeout: float = 5.0):
        self.root = Path(root).resolve()
        self.schemas = schemas
        self.manifest_timeout = manifest_timeout
        self.discovery_timeout = discovery_timeout
        self._items: dict[tuple[str, str], AdapterIdentity] = {}

    def discover(self) -> dict[tuple[str, str], AdapterIdentity]:
        found: dict[tuple[str, str], AdapterIdentity] = {}
        deadline=time.monotonic()+self.discovery_timeout
        for kind_dir in sorted(self.root.iterdir() if self.root.is_dir() else []):
            if not kind_dir.is_dir() or kind_dir.name not in VALID_KINDS:
                continue
            for impl_dir in sorted(p for p in kind_dir.iterdir() if p.is_dir()):
                entry = impl_dir / "adapter"
                if not entry.is_file() or not os.access(entry, os.X_OK):
                    continue
                remaining=deadline-time.monotonic()
                if remaining <= 0:
                    raise AdapterProtocolError("adapter discovery exceeded global bounded timeout")
                identity = self._load_identity(kind_dir.name, impl_dir.name, entry, manifest_timeout=min(self.manifest_timeout,remaining))
                key = (identity.kind, identity.name)
                if key in found:
                    raise ConfigError(f"duplicate adapter: {key[0]}/{key[1]}")
                found[key] = identity
        self._items = found
        return dict(found)

    def _load_identity(self, expected_kind: str, expected_name: str, entry: Path, *, manifest_timeout: float | None=None) -> AdapterIdentity:
        _validate_adapter_name(expected_name)
        manifest = self._read_manifest(entry, timeout=manifest_timeout)
        self.schemas.validate("adapter_manifest", manifest)
        _validate_adapter_name(str(manifest.get("name") or ""))
        _validate_schema_versions(manifest)
        if manifest["kind"] != expected_kind or manifest["name"] != expected_name:
            raise AdapterProtocolError(
                f"adapter location {expected_kind}/{expected_name} disagrees with manifest "
                f"{manifest['kind']}/{manifest['name']}"
            )
        if manifest["adapter_api"] != PROTOCOL_VERSION:
            raise AdapterProtocolError(
                f"unsupported adapter API {manifest['adapter_api']}; this release accepts exactly {PROTOCOL_VERSION}"
            )
        return AdapterIdentity(
            expected_kind, expected_name, manifest["version"], entry.resolve(), manifest,
        )

    def _read_manifest(self, entry: Path, *, timeout: float | None=None) -> dict:
        try:
            proc = subprocess.run(
                [str(entry), "manifest"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout or self.manifest_timeout, check=False, env=adapter_subprocess_env()
            )
        except subprocess.TimeoutExpired as exc:
            raise AdapterProtocolError(f"adapter manifest timed out: {entry}") from exc
        if proc.returncode != 0:
            raise AdapterProtocolError(f"adapter manifest failed: {entry}: {redact_text(proc.stderr.strip())}")
        try:
            return json_loads_strict(proc.stdout)
        except Exception as exc:
            raise AdapterProtocolError(f"adapter manifest is not JSON: {entry}") from exc

    def exists(self, kind: str, name: str) -> bool:
        if kind not in VALID_KINDS:
            return False
        try: _validate_adapter_name(name)
        except AdapterProtocolError: return False
        entry=self.root / kind / name / "adapter"
        return entry.is_file() and os.access(entry, os.X_OK)

    def get(self, kind: str, name: str) -> AdapterClient:
        key = (kind, name)
        identity = self._items.get(key)
        if identity is None:
            if kind not in VALID_KINDS:
                raise ConfigError(f"invalid adapter kind: {kind}")
            _validate_adapter_name(name)
            entry = self.root / kind / name / "adapter"
            if not entry.is_file() or not os.access(entry, os.X_OK):
                raise ConfigError(f"adapter not found: {kind}/{name}")
            identity = self._load_identity(kind, name, entry)
            self._items[key] = identity
        return AdapterClient(identity, self.schemas)

    def identities(self) -> list[AdapterIdentity]:
        if not self._items:
            self.discover()
        return [self._items[k] for k in sorted(self._items)]
