"""Stable console-script entry point.

The parser and command dispatch live in ``commands.cli_app`` so the installed
entry point remains small while command groups can evolve independently.
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
