from __future__ import annotations

from pathlib import Path

from jsonschema import Draft202012Validator

from model_evaluation.core.config.loader import reject_inline_secrets
from model_evaluation.core.config.parsing import load_json_strict, load_yaml_strict
from model_evaluation.core.errors import ConfigError
from model_evaluation.core.schema.formats import contract_format_checker


class MatrixSchemas:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _load(self, name: str) -> dict:
        path = self.root / f"{name}.schema.json"
        if not path.is_file():
            raise ConfigError(f"matrix schema missing: {path}")
        return load_json_strict(path.read_text(encoding="utf-8"))

    def validate(self, name: str, value: object) -> None:
        schema = self._load(name)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(
            schema,
            format_checker=contract_format_checker(),
        )
        errors = sorted(validator.iter_errors(value), key=lambda error: list(error.absolute_path))
        if errors:
            error = errors[0]
            path = ".".join(map(str, error.absolute_path)) or "<root>"
            raise ConfigError(f"{name} validation failed at {path}: {error.message}")


class MatrixRepository:
    def __init__(self, root: str | Path, schemas: MatrixSchemas):
        self.root = Path(root).resolve()
        self.schemas = schemas

    def load(self, value: str | Path) -> dict:
        path = Path(value)
        expected_id = None
        if not path.is_file():
            expected_id = str(value)
            if (
                not value
                or Path(value).is_absolute()
                or any(part in {"", ".", ".."} for part in Path(value).parts)
            ):
                raise ConfigError(f"invalid/path-escaping matrix spec id: {value!r}")
            candidates = [
                (self.root / f"{value}{extension}").resolve()
                for extension in (".yaml", ".yml", ".json")
            ]
            matches = [
                candidate
                for candidate in candidates
                if candidate.is_file()
                and (self.root == candidate.parent or self.root in candidate.parents)
            ]
            if len(matches) != 1:
                raise ConfigError(
                    f"expected exactly one matrix spec for {value!r}, found {len(matches)}"
                )
            path = matches[0]
        path = path.resolve()
        try:
            text = path.read_text(encoding="utf-8")
            value_object = (
                load_json_strict(text)
                if path.suffix.lower() == ".json"
                else load_yaml_strict(text)
            )
        except Exception as exc:
            raise ConfigError(f"failed to parse matrix spec {path}: {exc}") from exc
        if not isinstance(value_object, dict):
            raise ConfigError(f"matrix spec must be an object: {path}")
        reject_inline_secrets(value_object, str(path))
        self.schemas.validate("matrix_spec", value_object)
        if expected_id is not None and value_object.get("id") != expected_id:
            raise ConfigError(
                f"matrix spec filename/reference {expected_id!r} disagrees with id "
                f"{value_object.get('id')!r}"
            )
        return value_object
