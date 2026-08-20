#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


TOP_LEVEL_EXCLUDED = {
    "runtime",
    "results",
    "cache",
    "build",
    "dist",
    ".git",
    ".pytest_cache",
}
# These are operational conversion/verification tools.  They are part of the
# source/release ZIP so a model transformation can be repeated, but they are
# not imported by the installed runtime wheel.
SOURCE_TOOL_DIR = Path("scripts") / "model_conversion"
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}


def is_platform_metadata(path: Path) -> bool:
    """Return true for host metadata that is not a project input."""
    return path.name == ".DS_Store" or path.name.startswith("._")


def contains_platform_metadata(path: Path) -> bool:
    return any(is_platform_metadata(Path(part)) for part in path.parts)


def excluded_release_path(path: Path) -> bool:
    return (
        "__pycache__" in path.parts
        or contains_platform_metadata(path)
        or any(part.endswith(".egg-info") for part in path.parts)
        or (path.parts and path.parts[0] in TOP_LEVEL_EXCLUDED)
        or path.suffix in EXCLUDED_SUFFIXES
    )


def package_root(root: Path) -> Path:
    return root / "model_evaluation"


def sha256_file(path: Path) -> str:
    """Digest a release archive for transport verification only."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if excluded_release_path(path.relative_to(root)):
            continue
        yield path


def release_copy_ignore(root: Path):
    """Exclude generated/user state at the source-to-stage boundary."""
    project_root = root.resolve()

    def ignored(directory: str, names: list[str]) -> set[str]:
        current = Path(directory).resolve()
        omitted = {
            name
            for name in names
            if name == "__pycache__"
            or Path(name).suffix in EXCLUDED_SUFFIXES
            or is_platform_metadata(Path(name))
            or name.endswith(".egg-info")
        }
        if current == project_root:
            omitted.update(name for name in names if name in TOP_LEVEL_EXCLUDED)
        return omitted

    return ignored


def reject_release_symlinks(
    root: Path, *, label: str, ignore_excluded: bool = False
) -> None:
    """Do not dereference links into a release bundle."""
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if contains_platform_metadata(rel):
            continue
        if ignore_excluded and excluded_release_path(rel):
            continue
        if path.is_symlink():
            raise SystemExit(f"{label} contains symlink: {rel.as_posix()}")


def pep440(version: str) -> str:
    match = re.fullmatch(r"(\d+\.\d+\.\d+)-alpha(\d+)", version)
    if not match:
        raise SystemExit(f"unsupported VERSION.txt form: {version}")
    return f"{match.group(1)}a{match.group(2)}"


def check_version(root: Path) -> str:
    version = (package_root(root) / "VERSION.txt").read_text(encoding="utf-8").strip()
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    expected = pep440(version)
    if not match or match.group(1) != expected:
        actual = match.group(1) if match else None
        raise SystemExit(
            f"version mismatch: VERSION.txt={version}, pyproject={actual}, expected={expected}"
        )
    return version


def bounded(cmd: list[str], cwd: Path, timeout: float = 30.0):
    process = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if process.returncode:
        raise SystemExit(
            f"release gate failed: {' '.join(cmd)}\n"
            f"stdout:\n{process.stdout[-4000:]}\n"
            f"stderr:\n{process.stderr[-4000:]}"
        )
    return process


def validate_tree(root: Path) -> None:
    runtime_root = package_root(root)
    check_version(root)
    reject_release_symlinks(root, label="release tree", ignore_excluded=True)
    if any(path.name == "__pycache__" or path.suffix == ".pyc" for path in root.rglob("*")):
        raise SystemExit("compiled Python cache found in release")
    if any(contains_platform_metadata(path.relative_to(root)) for path in root.rglob("*")):
        raise SystemExit("platform metadata found in release")

    bounded([sys.executable, "tests/static_contract_check.py"], root)
    conversion_tools = root / SOURCE_TOOL_DIR
    if conversion_tools.is_dir():
        for tool in sorted(conversion_tools.glob("*.py")):
            bounded(
                [
                    sys.executable,
                    "-c",
                    "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read(), filename=sys.argv[1])",
                    str(tool.relative_to(root)),
                ],
                root,
            )
    if sys.platform.startswith("linux"):
        bounded(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
            root,
            timeout=240.0,
        )
    else:
        bounded(
            [
                sys.executable,
                "-m",
                "unittest",
                "tests.test_result_publication",
                "tests.test_result_product",
                "tests.test_batch_product",
            ],
            root,
            timeout=45.0,
        )

    smoke = (
        "from pathlib import Path; "
        "from model_evaluation.core.app import Application; "
        "r=Path.cwd(); a=Application(r/'model_evaluation',r); "
        "a.schemas.validate_all_schemas(); a.registry.discover(); "
        "p=a.plan('external_mmlu_example'); assert p['plan_id'].startswith('plan-')"
    )
    bounded([sys.executable, "-c", smoke], root)

    entries = list((runtime_root / "adapters").glob("*/*/adapter")) + [root / "eval-manager"]
    for entry in entries:
        if entry.is_file() and not os.access(entry, os.X_OK):
            raise SystemExit(f"executable mode missing: {entry.relative_to(root)}")


def check_installable_bundle(root: Path) -> None:
    """Build one wheel and verify the installed package-data tree."""
    with tempfile.TemporaryDirectory(prefix="model-eval-wheel-check-") as temp_dir:
        wheel_dir = Path(temp_dir) / "wheel"
        wheel_dir.mkdir()
        bounded(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-deps",
                "--no-build-isolation",
                "-w",
                str(wheel_dir),
            ],
            root,
            timeout=60.0,
        )
        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"wheel check expected one wheel, found: {wheels}")

        extracted = Path(temp_dir) / "installed"
        extracted.mkdir()
        with zipfile.ZipFile(wheels[0]) as archive:
            archive.extractall(extracted)
        installed = extracted / "model_evaluation"
        required = [
            installed / "cli.py",
            installed / "VERSION.txt",
            installed / "core" / "app.py",
            installed / "sdk" / "runtime.py",
            installed / "adapters" / "backend" / "vllm" / "adapter",
            installed / "schemas" / "backend_start_plan.schema.json",
            installed / "schemas" / "backend_preflight_plan.schema.json",
            installed / "schemas" / "preflight_probe_result.schema.json",
            installed / "schemas" / "preflight_report.schema.json",
            installed / "schemas" / "result.schema.json",
            installed / "schemas" / "metrics.schema.json",
            installed / "schemas" / "terminal.schema.json",
            installed / "schemas" / "failure.schema.json",
            installed / "presets" / "runs" / "external_mmlu_example.yaml",
        ]
        for item in required:
            if not item.exists():
                raise SystemExit(f"wheel runtime bundle missing: {item.relative_to(extracted)}")

        env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(extracted) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv=['eval-manager','schema-check']; "
                "from model_evaluation.cli import main; main()",
            ],
            cwd=extracted,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
            env=env,
        )
        if process.returncode or '"ok": true' not in process.stdout:
            raise SystemExit(
                "installed wheel schema-check failed\n"
                f"stdout:\n{process.stdout[-4000:]}\n"
                f"stderr:\n{process.stderr[-4000:]}"
            )

        # Extraction verifies package data, but it is not an installation test.
        # Install the built wheel into an isolated venv and exercise the actual
        # console script that users receive from pip.
        venv = Path(temp_dir) / "venv"
        bounded(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
            root,
            timeout=45.0,
        )
        scripts_dir = venv / ("Scripts" if os.name == "nt" else "bin")
        venv_python = scripts_dir / ("python.exe" if os.name == "nt" else "python")
        console = scripts_dir / ("eval-manager.exe" if os.name == "nt" else "eval-manager")

        # ``--system-site-packages`` points at the base interpreter, not at the
        # parent environment when the release gate itself runs inside a venv.
        # Make the already validated Controller dependencies visible without
        # installing from the network. The package under test is still installed
        # only from the newly built wheel.
        parent_sites = sorted(
            {
                str(Path(value).resolve())
                for value in sys.path
                if value
                and Path(value).is_dir()
                and Path(value).name in {"site-packages", "dist-packages"}
            }
        )
        purelib = bounded(
            [
                str(venv_python),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            root,
            timeout=20.0,
        ).stdout.strip()
        if parent_sites:
            Path(purelib, "model_eval_controller_dependencies.pth").write_text(
                "\n".join(parent_sites) + "\n",
                encoding="utf-8",
            )
        bounded(
            [str(venv_python), "-m", "pip", "install", "--no-deps", str(wheels[0])],
            root,
            timeout=45.0,
        )
        installed_schema = bounded([str(console), "schema-check"], root, timeout=20.0)
        if '"ok": true' not in installed_schema.stdout:
            raise SystemExit("pip-installed console script did not validate schemas")
        initialized = Path(temp_dir) / "initialized-project"
        bounded(
            [str(console), "init", str(initialized), "--hardware", "cpu"],
            root,
            timeout=20.0,
        )
        if not (initialized / "config" / "system.yaml").is_file():
            raise SystemExit("pip-installed init did not create a System config")

    shutil.rmtree(root / "build", ignore_errors=True)
    shutil.rmtree(root / "dist", ignore_errors=True)
    for path in root.glob("*.egg-info"):
        shutil.rmtree(path, ignore_errors=True)


def prepare_release_stage(root: Path, stage: Path) -> None:
    """Copy and validate the exact tree that will be archived."""
    shutil.copytree(root, stage, ignore=release_copy_ignore(root), symlinks=True)
    reject_release_symlinks(stage, label="release staging tree")
    validate_tree(stage)
    check_installable_bundle(stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    reject_release_symlinks(root, label="release source tree", ignore_excluded=True)
    version = check_version(root)
    release_root_name = f"model_evaluation_template_v{version.replace('-', '_')}"
    output = (
        Path(args.output).resolve()
        if args.output
        else root.parent / f"{release_root_name}.zip"
    )

    with tempfile.TemporaryDirectory(prefix="model-eval-release-") as temp_dir:
        stage_parent = Path(temp_dir) / "stage"
        stage_parent.mkdir()
        stage = stage_parent / release_root_name
        prepare_release_stage(root, stage)

        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files_under(stage):
                arcname = (Path(release_root_name) / path.relative_to(stage)).as_posix()
                archive.write(path, arcname=arcname)

        extract = Path(temp_dir) / "extract"
        extract.mkdir()
        with zipfile.ZipFile(output) as archive:
            for info in archive.infolist():
                target = Path(archive.extract(info, extract))
                mode = (info.external_attr >> 16) & 0o777
                if mode and target.exists():
                    os.chmod(target, mode)

            staged_runtime = package_root(stage)
            executable_paths = [
                Path(release_root_name) / "eval-manager",
                *[
                    Path(release_root_name) / path.relative_to(stage)
                    for path in staged_runtime.glob("adapters/*/*/adapter")
                ],
            ]
            for rel in executable_paths:
                info = archive.getinfo(rel.as_posix())
                if ((info.external_attr >> 16) & 0o111) == 0:
                    raise SystemExit(f"ZIP lost executable mode: {rel}")

        validate_tree(extract / release_root_name)

    print(
        json.dumps(
            {
                "ok": True,
                "zip": str(output),
                "transport_sha256": sha256_file(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
