from __future__ import annotations

from pathlib import Path
from jsonschema import Draft202012Validator, RefResolver
from jsonschema.exceptions import ValidationError

from model_evaluation.core.errors import SchemaValidationError
from model_evaluation.core.schema.formats import contract_format_checker
from model_evaluation.core.serialization import json_loads_strict

class SchemaStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise SchemaValidationError(f"schema directory not found: {self.root}")
        self._cache: dict[str, dict] = {}

    def load(self, name: str) -> dict:
        filename = name if name.endswith(".schema.json") else f"{name}.schema.json"
        if filename not in self._cache:
            path = self.root / filename
            if not path.is_file():
                raise SchemaValidationError(f"schema not found: {filename}")
            self._cache[filename] = json_loads_strict(path.read_text(encoding="utf-8"))
        return self._cache[filename]

    def _resolver_store(self) -> dict[str, dict]:
        """Preload package schemas under canonical and legacy resolution scopes.

        Supported jsonschema releases differ in how nested relative $id values
        update RefResolver scopes.  Keeping the small package schema set in an
        explicit store prevents filesystem or network fallback in every scope.
        """
        store: dict[str, dict] = {}
        resolution_roots = (self.root, *self.root.parents)
        for path in self.root.glob("*.schema.json"):
            document = self.load(path.name)
            for root in resolution_roots:
                store[(root / path.name).as_uri()] = document
            declared_id = document.get("$id")
            if isinstance(declared_id, str) and declared_id:
                store[declared_id] = document
        return store

    def _resolver(self, filename: str, schema: dict) -> RefResolver:
        return RefResolver(
            base_uri=(self.root / filename).as_uri(),
            referrer=schema,
            store=self._resolver_store(),
        )

    def validate(self, name: str, value: object) -> None:
        schema = self.load(name)
        filename = name if name.endswith(".schema.json") else f"{name}.schema.json"
        resolver = self._resolver(filename, schema)
        validator = Draft202012Validator(schema, resolver=resolver, format_checker=contract_format_checker())
        errors = sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path))
        if errors:
            err = errors[0]
            path = ".".join(str(x) for x in err.absolute_path) or "<root>"
            raise SchemaValidationError(f"{name} validation failed at {path}: {err.message}")


    def validate_def(self, name: str, def_name: str, value: object) -> None:
        schema = self.load(name)
        defs = schema.get("$defs") or {}
        if def_name not in defs:
            raise SchemaValidationError(f"schema definition not found: {name}#/$defs/{def_name}")
        filename = name if name.endswith(".schema.json") else f"{name}.schema.json"
        resolver = self._resolver(filename, schema)
        validator = Draft202012Validator(defs[def_name], resolver=resolver, format_checker=contract_format_checker())
        errors = sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path))
        if errors:
            err = errors[0]
            path = ".".join(str(x) for x in err.absolute_path) or "<root>"
            raise SchemaValidationError(f"{name}#/$defs/{def_name} validation failed at {path}: {err.message}")

    def validate_all_schemas(self) -> list[str]:
        checked = []
        for path in sorted(
            path
            for path in self.root.glob("*.schema.json")
            if not path.name.startswith("._")
        ):
            schema = self.load(path.name)
            Draft202012Validator.check_schema(schema)
            checked.append(path.name)
        return checked
