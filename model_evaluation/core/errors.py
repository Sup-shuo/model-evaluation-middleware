from __future__ import annotations

class ModelEvalError(RuntimeError):
    code = "MODEL_EVAL_ERROR"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}

class ConfigError(ModelEvalError): code = "CONFIG_INVALID"
class SchemaValidationError(ModelEvalError): code = "CONFIG_INVALID"
class AdapterProtocolError(ModelEvalError): code = "ADAPTER_PROTOCOL_ERROR"
class AdapterExecutionError(ModelEvalError):
    def __init__(self, code: str, message: str, *, retryable: bool = False, details: dict | None = None):
        super().__init__(message, details=details)
        self.code = code
        self.retryable = retryable
class CompatibilityError(ModelEvalError): code = "COMPATIBILITY_ERROR"
class ResourceError(ModelEvalError): code = "RESOURCE_UNAVAILABLE"
class ProcessError(ModelEvalError): code = "PROCESS_ERROR"
class CleanupCriticalError(ProcessError): code = "PROCESS_CLEANUP_CRITICAL"
class StaleProcessError(ProcessError): code = "PROCESS_STALE_OWNERSHIP"
class OrchestrationInterruptedError(ProcessError): code = "PROCESS_INTERRUPTED"
