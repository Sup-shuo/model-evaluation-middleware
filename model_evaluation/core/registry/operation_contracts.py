from __future__ import annotations

from model_evaluation.core.errors import AdapterProtocolError, SchemaValidationError
from model_evaluation.core.registry.operation_semantics import validate_output_semantics
from model_evaluation.core.schema.validator import SchemaStore


def validate_operation_input(
    schemas: SchemaStore,
    kind: str,
    operation: str,
    input_obj: dict,
) -> None:
    """Validate the Core-to-Adapter payload against the public RPC schema."""

    key = f"{kind}_{operation}"
    try:
        schemas.validate_def("adapter_operation_inputs", key, input_obj)
    except SchemaValidationError as exc:
        raise AdapterProtocolError(
            f"{kind}.{operation} input violates Adapter RPC contract: {exc}"
        ) from exc


def validate_operation_output(
    schemas: SchemaStore,
    kind: str,
    operation: str,
    output: dict,
    *,
    input_obj: dict | None = None,
) -> None:
    """Validate Adapter output Schema and cross-object semantics."""

    validate_output_semantics(
        schemas,
        kind,
        operation,
        output,
        input_obj=input_obj,
    )
