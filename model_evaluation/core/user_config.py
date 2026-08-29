from __future__ import annotations

from pathlib import Path

from model_evaluation.core.config.user_compiler import (
    UserConfigBundle,
    compile_user_config,
)


class UserConfigResolver:
    """Compile user-facing System and Evaluation YAML into internal Specs."""

    def __init__(self, app):
        self.app = app

    def load(
        self,
        system_path: str | Path,
        evaluation_path: str | Path,
        *,
        smoke: bool = False,
    ) -> UserConfigBundle:
        return compile_user_config(
            self.app,
            system_path,
            evaluation_path,
            smoke=smoke,
        )


__all__ = ["UserConfigBundle", "UserConfigResolver"]
