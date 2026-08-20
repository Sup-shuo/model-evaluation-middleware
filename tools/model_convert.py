#!/usr/bin/env python3
"""Manual, non-destructive entry point for checkpoint conversion utilities.

This command is intentionally separate from eval-manager.  Evaluation,
validation, Doctor and planning never invoke it automatically.  A conversion
only starts after the operator selects an explicit route and runs ``convert``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMPLEMENTATION_ROOT = Path(__file__).resolve().parent / "model_conversion"


@dataclass(frozen=True)
class Route:
    name: str
    script: str
    description: str
    architectures: tuple[str, ...]
    quantization_method: str | None
    contract_required: bool = False
    independent_verifier: bool = False


ROUTES = {
    route.name: route
    for route in (
        Route(
            name="compressed-tensors-to-bf16",
            script="convert_ct_qwen_to_dense_bf16.py",
            description=(
                "Dequantize supported Qwen compressed-tensors W4A16 weights "
                "to a separate dense BF16 checkpoint."
            ),
            architectures=(
                "Qwen3_5ForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
            ),
            quantization_method="compressed-tensors",
            independent_verifier=True,
        ),
        Route(
            name="awq-to-bf16",
            script="convert_awq_qwen_to_dense_bf16.py",
            description=(
                "Dequantize the supported Qwen3.5 AWQ GEMM W4 group-128 "
                "layout to a separate dense BF16 checkpoint."
            ),
            architectures=("Qwen3_5ForConditionalGeneration",),
            quantization_method="awq",
        ),
        Route(
            name="compressed-tensors-contract-to-bf16",
            script="convert_ct_with_contract_bf16.py",
            description=(
                "Dequantize Qwen3.5 compressed-tensors weights using an "
                "explicit, independently prepared output contract."
            ),
            architectures=("Qwen3_5ForConditionalGeneration",),
            quantization_method="compressed-tensors",
            contract_required=True,
        ),
        Route(
            name="qwen3vl-text-to-bf16",
            script="extract_qwen3vl_text_bf16.py",
            description=(
                "Extract the text-only Qwen3 language model from a dense "
                "Qwen3-VL BF16 checkpoint; visual capability is removed."
            ),
            architectures=("Qwen3VLForConditionalGeneration",),
            quantization_method=None,
        ),
    )
}


def load_config(source: Path) -> dict[str, Any]:
    config_path = source / "config.json"
    if not source.is_dir():
        raise ValueError(f"source is not a directory: {source}")
    if not config_path.is_file():
        raise ValueError(f"config.json is missing: {config_path}")
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read config.json: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("config.json must contain a JSON object")
    return value


def inspect_checkpoint(source: Path) -> dict[str, Any]:
    source = source.resolve()
    config = load_config(source)
    architectures = config.get("architectures") or []
    architecture = architectures[0] if len(architectures) == 1 else None
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        quantization = {}
    quantization_method = quantization.get("quant_method")

    compatible_routes = []
    for route in ROUTES.values():
        architecture_ok = architecture in route.architectures
        quantization_ok = (
            quantization_method == route.quantization_method
            if route.quantization_method is not None
            else not quantization
        )
        if architecture_ok and quantization_ok:
            compatible_routes.append(route.name)

    return {
        "source": str(source),
        "architecture": architecture,
        "model_type": config.get("model_type"),
        "dtype": config.get("dtype") or config.get("torch_dtype"),
        "quantization": {
            key: quantization.get(key)
            for key in ("quant_method", "format", "bits", "group_size", "version")
            if quantization.get(key) is not None
        },
        "compatible_manual_routes": compatible_routes,
        "automatic_conversion": False,
    }


def validate_route(route: Route, inspection: dict[str, Any]) -> None:
    architecture = inspection["architecture"]
    quantization_method = inspection["quantization"].get("quant_method")
    if architecture not in route.architectures:
        raise ValueError(
            f"route {route.name!r} does not support architecture {architecture!r}"
        )
    if route.quantization_method is None:
        if quantization_method is not None:
            raise ValueError(
                f"route {route.name!r} requires a dense source, found "
                f"quantization_method={quantization_method!r}"
            )
    elif quantization_method != route.quantization_method:
        raise ValueError(
            f"route {route.name!r} requires quantization_method="
            f"{route.quantization_method!r}, found {quantization_method!r}"
        )


def route_payload(route: Route) -> dict[str, Any]:
    return {
        "name": route.name,
        "description": route.description,
        "architectures": list(route.architectures),
        "quantization_method": route.quantization_method,
        "contract_required": route.contract_required,
        "independent_verifier": route.independent_verifier,
    }


def print_value(value: Any, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if isinstance(value, list):
        for item in value:
            print(f"{item['name']}: {item['description']}")
        return
    print(f"source: {value['source']}")
    print(f"architecture: {value['architecture'] or 'unknown'}")
    print(f"model_type: {value['model_type'] or 'unknown'}")
    print(f"dtype: {value['dtype'] or 'unknown'}")
    print(f"quantization: {json.dumps(value['quantization'], ensure_ascii=False)}")
    routes = value["compatible_manual_routes"]
    print("compatible manual routes: " + (", ".join(routes) if routes else "none"))
    print("automatic conversion: disabled")


def conversion_command(args: argparse.Namespace) -> list[str]:
    route = ROUTES[args.route]
    source = args.source.resolve()
    output = args.output.absolute()
    temporary = (
        args.temporary.absolute()
        if args.temporary is not None
        else output.with_name(f".{output.name}.converting")
    )
    inspection = inspect_checkpoint(source)
    validate_route(route, inspection)
    if source == output.resolve(strict=False) or source == temporary.resolve(strict=False):
        raise ValueError("source, temporary and output directories must be distinct")
    if temporary.parent != output.parent:
        raise ValueError("temporary and output directories must share a parent")
    if not output.parent.is_dir():
        raise ValueError(f"output parent does not exist: {output.parent}")
    if output.exists() or temporary.exists():
        raise ValueError(
            f"refusing to overwrite existing output/temporary path: {output} / {temporary}"
        )
    if route.contract_required and args.contract is None:
        raise ValueError(f"route {route.name!r} requires --contract")
    if not route.contract_required and args.contract is not None:
        raise ValueError(f"route {route.name!r} does not accept --contract")

    command = [
        str(args.python),
        str(IMPLEMENTATION_ROOT / route.script),
        "--source",
        str(source),
        "--temporary",
        str(temporary),
        "--final",
        str(output),
        "--source-ref",
        args.source_ref,
    ]
    if args.contract is not None:
        command.extend(["--contract", str(args.contract.resolve())])
    if route.name == "compressed-tensors-contract-to-bf16":
        command.extend(["--max-shard-bytes", str(args.max_shard_bytes)])
    else:
        command.extend(["--max-shard-size", args.max_shard_size])
    return command


def verify_command(args: argparse.Namespace) -> list[str]:
    route = ROUTES[args.route]
    if not route.independent_verifier:
        raise ValueError(
            f"route {route.name!r} has no separate verifier; its converter performs "
            "full validation before publishing"
        )
    inspection = inspect_checkpoint(args.source.resolve())
    validate_route(route, inspection)
    if not args.derived.is_dir():
        raise ValueError(f"derived checkpoint is not a directory: {args.derived}")
    return [
        str(args.python),
        str(IMPLEMENTATION_ROOT / "verify_dense_qwen_checkpoint.py"),
        "--source",
        str(args.source.resolve()),
        "--derived",
        str(args.derived.resolve()),
        "--expected-parameters",
        str(args.expected_parameters),
    ]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=(
            "Inspect, manually convert and verify model checkpoints. "
            "This tool is never called automatically by eval-manager."
        )
    )
    commands = root.add_subparsers(dest="command", required=True)

    routes = commands.add_parser("routes", help="list explicit conversion routes")
    routes.add_argument("--format", choices=("text", "json"), default="text")

    inspect = commands.add_parser("inspect", help="read checkpoint metadata without conversion")
    inspect.add_argument("source", type=Path)
    inspect.add_argument("--format", choices=("text", "json"), default="text")

    convert = commands.add_parser("convert", help="run one explicitly selected route")
    convert.add_argument("--route", choices=tuple(ROUTES), required=True)
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--temporary", type=Path)
    convert.add_argument("--source-ref", required=True)
    convert.add_argument("--contract", type=Path)
    convert.add_argument("--max-shard-size", default="5GB")
    convert.add_argument("--max-shard-bytes", type=int, default=4_000_000_000)
    convert.add_argument("--python", type=Path, default=Path(sys.executable))
    convert.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact implementation command without starting conversion",
    )

    verify = commands.add_parser("verify", help="independently verify a published derivative")
    verify.add_argument("--route", choices=tuple(ROUTES), required=True)
    verify.add_argument("--source", type=Path, required=True)
    verify.add_argument("--derived", type=Path, required=True)
    verify.add_argument("--expected-parameters", type=int, required=True)
    verify.add_argument("--python", type=Path, default=Path(sys.executable))
    verify.add_argument("--dry-run", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "routes":
            print_value([route_payload(route) for route in ROUTES.values()], args.format)
            return 0
        if args.command == "inspect":
            print_value(inspect_checkpoint(args.source), args.format)
            return 0
        command = conversion_command(args) if args.command == "convert" else verify_command(args)
        if args.dry_run:
            print(json.dumps({"command": command, "automatic_conversion": False}, indent=2))
            return 0
        print("manual conversion command:", " ".join(command), flush=True)
        return subprocess.run(command, check=False).returncode
    except (OSError, ValueError) as error:
        print(f"model-convert: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
