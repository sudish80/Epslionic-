__test__ = False


class OpenClawError(Exception):
    """Base exception for all OpenClaw agent errors."""
    code = "OPENCLAW_ERROR"

    def __init__(self, message="", cause=None, context=None):
        self.message = message
        self.cause = cause
        self.context = context or {}
        super().__init__(self.message)
        if cause:
            self.__cause__ = cause if isinstance(cause, BaseException) else None

    def __str__(self):
        parts = [self.message]
        if self.cause:
            parts.append(f"[caused by: {self.cause}]")
        if self.context:
            ctx = "; ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"({ctx})")
        return " ".join(parts)


class ConfigurationError(OpenClawError):
    """Invalid configuration or missing required settings."""
    code = "CONFIG_ERROR"


class DomainError(OpenClawError):
    """Domain selection or validation error."""
    code = "DOMAIN_ERROR"


class EnvironmentError(OpenClawError):
    """Hardware or environment issue (no GPU, no internet, etc.)."""
    code = "ENV_ERROR"


class GPUError(EnvironmentError):
    """GPU-related error (CUDA unavailable, out of memory, etc.)."""
    code = "GPU_ERROR"


class NetworkError(OpenClawError):
    """Network or API connectivity error."""
    code = "NETWORK_ERROR"


class DatasetError(OpenClawError):
    """Dataset discovery, loading, or processing error."""
    code = "DATASET_ERROR"


class TrainingError(OpenClawError):
    """Training execution error."""
    code = "TRAINING_ERROR"


class ModelError(OpenClawError):
    """Model loading, saving, or configuration error."""
    code = "MODEL_ERROR"


class MemoryError(OpenClawError):
    """Memory store read/write error."""
    code = "MEMORY_ERROR"


class ToolError(OpenClawError):
    """Tool execution error."""
    code = "TOOL_ERROR"


class HeartbeatError(OpenClawError):
    """Heartbeat monitoring error."""
    code = "HEARTBEAT_ERROR"


class AuthenticationError(OpenClawError):
    """API key or token authentication error."""
    code = "AUTH_ERROR"


class BrainError(OpenClawError):
    """LLM Brain reasoning or query error."""
    code = "BRAIN_ERROR"


ERROR_CODES = {
    ConfigurationError: "CONFIG_ERROR",
    DomainError: "DOMAIN_ERROR",
    GPUError: "GPU_ERROR",
    NetworkError: "NETWORK_ERROR",
    DatasetError: "DATASET_ERROR",
    TrainingError: "TRAINING_ERROR",
    AuthenticationError: "AUTH_ERROR",
    BrainError: "BRAIN_ERROR",
    MemoryError: "MEMORY_ERROR",
    ToolError: "TOOL_ERROR",
    HeartbeatError: "HEARTBEAT_ERROR",
}


def error_code(exc: Exception) -> str:
    """Get a stable error code for any exception."""
    for exc_type, code in ERROR_CODES.items():
        if isinstance(exc, exc_type):
            return code
    return "UNKNOWN_ERROR"


def as_error(exc: BaseException) -> OpenClawError:
    """Convert any exception to an OpenClawError with cause preserved."""
    if isinstance(exc, OpenClawError):
        return exc
    return OpenClawError(message=str(exc), cause=exc)
