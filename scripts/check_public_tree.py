#!/usr/bin/env python3
"""Fail when a public source tree contains likely private deployment data."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL_EXCLUDED = {".git", "results", "cache", "runtime", "build", "dist"}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PATTERNS = {
    "macOS user home": re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    "Linux user home": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "personal AFS path": re.compile(
        r"/(?:root|mnt)/(?:[^\s'\"`]+/)*afs/(?:users/)?[A-Za-z0-9._-]+/"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[opurs]_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "non-loopback IPv4": re.compile(
        r"(?<![\d.])(?!(?:127\.0\.0\.1|0\.0\.0\.0)(?![\d.]))"
        r"(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"
    ),
}


def files() -> list[Path]:
    rows = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file():
            continue
        if relative.parts and relative.parts[0] in TOP_LEVEL_EXCLUDED:
            continue
        if "__pycache__" in relative.parts:
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            rows.append(path)
    return sorted(rows)


def main() -> None:
    findings: list[str] = []
    public_files = files()
    for path in public_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{path.relative_to(ROOT)}:{line_number}: {label}")
    if findings:
        raise SystemExit("public-tree privacy check failed:\n" + "\n".join(findings))
    print(f"public-tree privacy check passed: {len(public_files)} text files")


if __name__ == "__main__":
    main()
