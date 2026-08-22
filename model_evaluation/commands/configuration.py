from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from model_evaluation.core.config.catalog import (
    CONFIG_KINDS,
    scan_config_catalog,
)
from model_evaluation.core.config.migration import migrate_entries, migrate_entry
from model_evaluation.core.errors import ConfigError


def add_config_commands(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("config", help="列出、检查、查看和迁移用户配置目录")
    actions = parser.add_subparsers(dest="config_action", required=True)

    listing = actions.add_parser("list", help="列出配置目录中的 System、Model 和 Evaluation")
    listing.add_argument("--kind", choices=CONFIG_KINDS)
    listing.add_argument("--format", choices=("human", "json"), default="human")

    showing = actions.add_parser("show", help="显示一份原始配置；最终解析结果请使用 plan")
    showing.add_argument("kind", choices=CONFIG_KINDS)
    showing.add_argument("reference")
    showing.add_argument("--format", choices=("yaml", "json"), default="yaml")

    checking = actions.add_parser("check", help="检查整个配置目录及可选的 System/Evaluation 组合")
    checking.add_argument("--kind", choices=CONFIG_KINDS)
    checking.add_argument("--system-config")
    checking.add_argument("--evaluation-config")
    checking.add_argument("--format", choices=("human", "json"), default="human")

    migrating = actions.add_parser("migrate", help="预览或写入受支持的用户 Schema 迁移")
    migrating.add_argument("--kind", choices=("system", "evaluation"))
    migrating.add_argument("--write", action="store_true")
    migrating.add_argument("--backend-profile")
    migrating.add_argument("--evaluator-profile")
    migrating.add_argument("--format", choices=("human", "json"), default="human")


def _emit(payload: dict, *, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if "entries" in payload:
        for row in payload["entries"]:
            suffix = f" ERROR: {row['error']}" if row.get("error") else ""
            print(f"{row['kind']:<10} {row['reference']:<36} schema={row.get('schema_version') or '-'}{suffix}")
    if "migrations" in payload:
        for row in payload["migrations"]:
            state = "written" if row.get("written") else ("pending" if row.get("changed") else "current")
            print(f"{row['kind']:<10} {row['reference']:<36} {row['from']} -> {row['to']} {state}")
    for row in payload.get("errors") or []:
        print(f"ERROR {row.get('kind', '-')}/{row.get('reference', '-')}: {row['error']}")
    print(f"Result: {'OK' if payload.get('ok') else 'FAILED'}")


def _entry_rows(entries) -> list[dict]:
    return [
        {
            "kind": entry.kind,
            "reference": entry.reference,
            "path": str(entry.path),
            "schema_version": entry.schema_version,
            "error": entry.error,
        }
        for entry in entries
    ]


def _find_entry(app, kind: str, reference: str):
    matches = [
        entry
        for entry in scan_config_catalog(app.project_root, kind)
        if entry.reference == reference
    ]
    if len(matches) != 1:
        raise ConfigError(f"expected exactly one {kind} configuration for {reference!r}, found {len(matches)}")
    if matches[0].error or matches[0].data is None:
        raise ConfigError(matches[0].error or "invalid configuration")
    return matches[0]


def _check_catalog(app, kind: str | None) -> dict:
    entries = scan_config_catalog(app.project_root, kind)
    rows = _entry_rows(entries)
    counts = Counter((entry.kind, entry.reference) for entry in entries)
    duplicates = {key for key, count in counts.items() if count > 1}
    schema_names = {"system": "user_system", "model": "user_model", "evaluation": "user_evaluation"}
    ok = True
    for entry, row in zip(entries, rows):
        if (entry.kind, entry.reference) in duplicates:
            row["error"] = f"duplicate {entry.kind} reference: {entry.reference}"
        if not row.get("error"):
            try:
                app.matrix_schemas.validate(schema_names[entry.kind], entry.data)
            except Exception as exc:
                row["error"] = str(exc)
        ok = ok and not bool(row.get("error"))
    return {"ok": ok, "entries": rows, "count": len(rows)}


def handle_config_command(args: argparse.Namespace, app) -> bool:
    if args.cmd != "config":
        return False
    if args.config_action == "list":
        entries = scan_config_catalog(app.project_root, args.kind)
        payload = {"ok": not any(entry.error for entry in entries), "entries": _entry_rows(entries), "count": len(entries)}
        _emit(payload, output_format=args.format)
        if not payload["ok"]:
            raise SystemExit(2)
        return True
    if args.config_action == "show":
        entry = _find_entry(app, args.kind, args.reference)
        if args.format == "json":
            print(json.dumps(entry.data, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(yaml.safe_dump(entry.data, allow_unicode=True, sort_keys=False), end="")
        return True
    if args.config_action == "check":
        payload = _check_catalog(app, args.kind)
        if args.system_config or args.evaluation_config:
            if not (args.system_config and args.evaluation_config):
                raise ConfigError("config check pair validation requires both --system-config and --evaluation-config")
            try:
                bundle = app.load_user_config(args.system_config, args.evaluation_config)
                payload["pair"] = {
                    "ok": True,
                    "system": bundle.system["system"]["name"],
                    "profiles": bundle.generated.get("selected_profiles", {}),
                }
            except Exception as exc:
                payload["pair"] = {"ok": False, "error": str(exc)}
                payload["ok"] = False
        _emit(payload, output_format=args.format)
        if not payload["ok"]:
            raise SystemExit(2)
        return True
    if args.config_action == "migrate":
        entries = scan_config_catalog(app.project_root, args.kind)
        migrations = []
        errors = []
        migration_entries = [
            entry for entry in entries if entry.kind in {"system", "evaluation"}
        ]
        for entry in migration_entries:
            try:
                migrations.append(
                    migrate_entry(
                        entry,
                        write=False,
                        backend_profile=args.backend_profile,
                        evaluator_profile=args.evaluator_profile,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "kind": entry.kind,
                        "reference": entry.reference,
                        "path": str(entry.path),
                        "error": str(exc),
                    }
                )
        if args.write and not errors:
            try:
                migrations = migrate_entries(
                    migration_entries,
                    write=True,
                    backend_profile=args.backend_profile,
                    evaluator_profile=args.evaluator_profile,
                )
            except Exception as exc:
                errors.append(
                    {
                        "kind": args.kind or "catalog",
                        "reference": "*",
                        "path": str(app.project_root / "config"),
                        "error": str(exc),
                    }
                )
        payload = {"ok": not errors, "write": bool(args.write), "migrations": migrations, "errors": errors}
        _emit(payload, output_format=args.format)
        if errors:
            raise SystemExit(2)
        return True
    return False
