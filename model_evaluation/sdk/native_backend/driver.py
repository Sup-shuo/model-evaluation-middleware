from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class NativeDriverError(RuntimeError):
    """A model-driver failure safe to expose through the service protocol."""

    def __init__(self, code: str, message: str, *, status: int = 400):
        super().__init__(message)
        self.code = str(code or "NATIVE_DRIVER_ERROR")
        self.status = int(status)


class NativeDriver(ABC):
    """Minimal model-family contract hosted by the shared native service."""

    @abstractmethod
    def load(self) -> None:
        """Load the model in the selected Backend Environment."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Describe the loaded model and truthful runtime capabilities."""

    def completion(self, prompt: str, parameters: dict[str, Any]) -> dict[str, Any]:
        raise NativeDriverError(
            "OPERATION_UNSUPPORTED",
            "this driver does not provide text completions",
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        raise NativeDriverError(
            "OPERATION_UNSUPPORTED",
            "this driver does not provide chat generation",
        )

    def tokenize(self, text: str) -> list[int]:
        raise NativeDriverError(
            "OPERATION_UNSUPPORTED",
            "this driver does not provide tokenization",
        )

    def detokenize(self, tokens: list[int]) -> str:
        raise NativeDriverError(
            "OPERATION_UNSUPPORTED",
            "this driver does not provide detokenization",
        )

    def close(self) -> None:
        """Release model resources. Drivers may override this hook."""
