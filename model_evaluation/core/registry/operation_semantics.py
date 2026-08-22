from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from urllib.parse import urlsplit

from model_evaluation.core.errors import AdapterProtocolError
from model_evaluation.core.schema.validator import SchemaStore


_HEX64 = re.compile(r"^[a-f0-9]{64}$")
OutputValidator = Callable[[SchemaStore, str, dict, dict, str], None]


def _obj(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise AdapterProtocolError(f"{label} must be an object")
    return value


def _require(obj: Mapping, key: str, label: str):
    if key not in obj:
        raise AdapterProtocolError(f"{label} missing required field {key!r}")
    return obj[key]


def _validate_requirement_namespaces(
    requirement_set: dict,
    *,
    allowed_prefixes: tuple[str, ...],
    label: str,
) -> None:
    for requirement in requirement_set.get("requirements", []):
        path = requirement.get("path")
        if not isinstance(path, str) or not path.startswith(allowed_prefixes):
            raise AdapterProtocolError(
                f"{label} requirement path {path!r} escapes allowed canonical "
                f"namespaces {allowed_prefixes}"
            )


def _validate_auth_semantics(auth: dict) -> None:
    mode = auth.get("mode")
    reference = auth.get("secret_ref")
    if mode == "none" and reference is not None:
        raise AdapterProtocolError("auth.mode=none may not carry secret_ref")
    if mode == "bearer" and not isinstance(reference, str):
        raise AdapterProtocolError("auth.mode=bearer requires secret_ref")
    if mode == "custom":
        raise AdapterProtocolError(
            "auth.mode=custom is reserved by Protocol v1 and is not implemented "
            "by Adapter Protocol v1 in this release"
        )


def _validate_attach_semantics(attach: dict, input_obj: dict, label: str) -> None:
    base_url = str(attach.get("base_url") or "").strip()
    try:
        parsed = urlsplit(base_url)
    except Exception as exc:
        raise AdapterProtocolError(f"{label}.attach.base_url is invalid") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise AdapterProtocolError(
            f"{label}.attach.base_url must be an absolute http/https URL"
        )
    if parsed.username is not None or parsed.password is not None:
        raise AdapterProtocolError(
            f"{label}.attach.base_url may not embed credentials"
        )
    if parsed.query or parsed.fragment:
        raise AdapterProtocolError(
            f"{label}.attach.base_url may not contain query or fragment"
        )

    model_id = attach.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise AdapterProtocolError(f"{label}.attach.model_id must be non-empty")
    ownership = attach.get("ownership")
    if ownership not in {"managed", "attached", "external"}:
        raise AdapterProtocolError(
            f"{label}.attach.ownership is invalid: {ownership!r}"
        )
    declared = ((input_obj.get("deployment") or {}).get("management") or {}).get(
        "mode"
    )
    if declared and ownership != declared:
        raise AdapterProtocolError(
            f"{label}.attach.ownership {ownership!r} disagrees with "
            f"DeploymentProfile mode {declared!r}"
        )
    if ownership == "managed":
        expected_port = (input_obj.get("endpoint") or {}).get("port")
        if expected_port is not None:
            try:
                actual_port = parsed.port
            except ValueError as exc:
                raise AdapterProtocolError(
                    f"{label}.attach.base_url has an invalid port"
                ) from exc
            if actual_port != int(expected_port):
                raise AdapterProtocolError(
                    f"{label}.attach.base_url port {actual_port!r} disagrees with "
                    f"planned endpoint port {int(expected_port)!r}"
                )
    _validate_auth_semantics(_obj(attach.get("auth"), f"{label}.attach.auth"))


def _validate_device(schemas, operation, output, input_obj, label) -> None:
    if operation == "probe":
        schemas.validate("device_descriptor", output)
    elif operation == "visibility":
        schemas.validate(
            "env_patch",
            _obj(_require(output, "env_patch", label), f"{label}.env_patch"),
        )


def _validate_runtime(schemas, operation, output, input_obj, label) -> None:
    if operation == "probe":
        schemas.validate("runtime_descriptor", output)
    elif operation == "resolve_environment":
        schemas.validate(
            "env_patch",
            _obj(_require(output, "env_patch", label), f"{label}.env_patch"),
        )


def _validate_environment(schemas, operation, output, input_obj, label) -> None:
    if operation == "resolve":
        schemas.validate("environment_descriptor", output)
    elif operation == "wrap_process":
        schemas.validate(
            "process_spec",
            _obj(_require(output, "process", label), f"{label}.process"),
        )


def _validate_backend(schemas, operation, output, input_obj, label) -> None:
    if operation == "requirements":
        schemas.validate("requirement_set", output)
        _validate_requirement_namespaces(
            output,
            allowed_prefixes=("device.", "runtime.", "backend_environment."),
            label=label,
        )
    elif operation == "plan_preflight":
        schemas.validate("backend_preflight_plan", output)
        probes = output.get("probes", [])
        probe_ids = [str(row.get("id")) for row in probes]
        if len(probe_ids) != len(set(probe_ids)):
            raise AdapterProtocolError(f"{label} probe ids must be unique")
        if not any(
            row.get("phase") == "backend_dependency" and row.get("required")
            for row in probes
        ):
            raise AdapterProtocolError(
                f"{label} requires at least one required backend_dependency probe"
            )
        seen_model_phase = False
        for row in probes:
            if row.get("phase") == "model_compatibility":
                seen_model_phase = True
            elif seen_model_phase:
                raise AdapterProtocolError(
                    f"{label} backend_dependency probes must precede "
                    "model_compatibility probes"
                )
    elif operation == "plan_start":
        schemas.validate("backend_start_plan", output)
        attach = _obj(_require(output, "attach", label), f"{label}.attach")
        for key in ("base_url", "model_id", "ownership", "auth"):
            _require(attach, key, f"{label}.attach")
        _validate_attach_semantics(attach, input_obj, label)
        if "process" in output:
            schemas.validate(
                "process_spec", _obj(output["process"], f"{label}.process")
            )
        managed = attach["ownership"] == "managed"
        for field in ("process", "dependency_probe", "shutdown"):
            if managed and field not in output:
                raise AdapterProtocolError(
                    f"{label} managed ownership requires {field}"
                )
            if not managed and field in output:
                raise AdapterProtocolError(
                    f"{label} non-managed ownership may not return {field}"
                )
        if "readiness" in output:
            readiness = _obj(output["readiness"], f"{label}.readiness")
            if "timeout_seconds" in readiness:
                value = readiness["timeout_seconds"]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value <= 0
                ):
                    raise AdapterProtocolError(
                        f"{label}.readiness.timeout_seconds must be positive numeric"
                    )
    elif operation == "probe_service":
        schemas.validate("service_descriptor", output)
        _validate_auth_semantics(output.get("auth") or {})
        attach = _obj(input_obj.get("attach"), f"{label}.input.attach")
        if (output.get("model") or {}).get("id") != attach.get("model_id"):
            raise AdapterProtocolError(
                f"{label} model.id disagrees with attach.model_id"
            )
        if output.get("ownership") != attach.get("ownership"):
            raise AdapterProtocolError(
                f"{label} ownership disagrees with attach.ownership"
            )
        expected_auth = attach.get("auth") or {"mode": "none"}
        actual_auth = output.get("auth") or {"mode": "none"}
        for key in ("mode", "secret_ref"):
            if actual_auth.get(key) != expected_auth.get(key):
                raise AdapterProtocolError(
                    f"{label} auth.{key} disagrees with attach.auth.{key}"
                )


def _validate_dataset(schemas, operation, output, input_obj, label) -> None:
    if operation == "prepare":
        schemas.validate("dataset_artifact", output)
    elif operation == "verify":
        valid = _require(output, "valid", label)
        if not isinstance(valid, bool):
            raise AdapterProtocolError(f"{label}.valid must be boolean")
        if "artifact" in output:
            schemas.validate(
                "dataset_artifact", _obj(output["artifact"], f"{label}.artifact")
            )


def _validate_binding(schemas, operation, output, input_obj, label) -> None:
    if operation == "requirements":
        schemas.validate("requirement_set", output)
        _validate_requirement_namespaces(
            output,
            allowed_prefixes=(
                "device.",
                "runtime.",
                "backend_environment.",
                "evaluation_environment.",
            ),
            label=label,
        )
    elif operation == "build_task":
        schemas.validate("framework_task_artifact", output)
        benchmark = input_obj.get("benchmark") or {}
        evaluation = input_obj.get("evaluation") or {}
        expected_benchmark = benchmark.get("id")
        expected_framework = (evaluation.get("framework") or {}).get("adapter")
        if expected_benchmark and output.get("benchmark_id") != expected_benchmark:
            raise AdapterProtocolError(
                f"{label} benchmark_id disagrees with BenchmarkSpec.id"
            )
        if expected_framework and output.get("framework") != expected_framework:
            raise AdapterProtocolError(
                f"{label} framework disagrees with EvaluationProfile framework adapter"
            )
    elif operation == "protocol_fingerprint":
        value = _require(output, "protocol_fingerprint", label)
        if not isinstance(value, str) or not _HEX64.fullmatch(value):
            raise AdapterProtocolError(
                f"{label}.protocol_fingerprint must be sha256 hex"
            )


def _validate_evaluator(schemas, operation, output, input_obj, label) -> None:
    if operation == "requirements":
        schemas.validate("requirement_set", output)
        _validate_requirement_namespaces(
            output,
            allowed_prefixes=("service.", "evaluation_environment."),
            label=label,
        )
    elif operation == "plan_preflight":
        schemas.validate(
            "process_spec",
            _obj(_require(output, "process", label), f"{label}.process"),
        )
        unknown = set(output) - {"process", "result_format"}
        if unknown:
            raise AdapterProtocolError(
                f"{label} contains unknown fields: {sorted(unknown)}"
            )
        result_format = output.get("result_format", "text")
        if result_format not in {"text", "preflight_result"}:
            raise AdapterProtocolError(
                f"{label}.result_format must be text or preflight_result"
            )
    elif operation == "plan_evaluate":
        schemas.validate(
            "process_spec",
            _obj(_require(output, "process", label), f"{label}.process"),
        )
        root = _require(output, "raw_result_root", label)
        if not isinstance(root, str) or not root:
            raise AdapterProtocolError(
                f"{label}.raw_result_root must be non-empty string"
            )
    elif operation == "normalize":
        schemas.validate("canonical_result", output)


OUTPUT_VALIDATORS: dict[str, OutputValidator] = {
    "device": _validate_device,
    "runtime": _validate_runtime,
    "environment": _validate_environment,
    "backend": _validate_backend,
    "dataset": _validate_dataset,
    "binding": _validate_binding,
    "evaluator": _validate_evaluator,
}


def validate_output_semantics(
    schemas: SchemaStore,
    kind: str,
    operation: str,
    output: dict,
    *,
    input_obj: dict | None = None,
) -> None:
    label = f"{kind}.{operation} output"
    _obj(output, label)
    validator = OUTPUT_VALIDATORS.get(kind)
    if validator is not None:
        validator(schemas, operation, output, input_obj or {}, label)
