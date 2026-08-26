"""Stable console-script entry point.

Argument construction lives in ``commands.parser`` and command dispatch in
``commands.cli_app`` so the installed entry point stays small and stable.
"""

from model_evaluation.commands.cli_app import (
    EVALUATOR_DOCTOR_RPC_TIMEOUT_SECONDS,
    doctor_dump,
    main,
)

__all__ = [
    "EVALUATOR_DOCTOR_RPC_TIMEOUT_SECONDS",
    "doctor_dump",
    "main",
]


if __name__ == "__main__":
    main()
