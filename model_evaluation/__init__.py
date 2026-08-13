"""Model evaluation middleware package."""

from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    """Return the installed package-data root."""
    return Path(__file__).resolve().parent


__all__ = ["package_root"]
