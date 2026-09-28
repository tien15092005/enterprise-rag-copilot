from __future__ import annotations

from typing import Any

from app.core.exceptions import AppError


class AgentError(AppError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "agent_error",
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, status_code=status_code, details=details)


class UnknownAgentToolError(AgentError):
    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"agent tool '{tool_name}' is not allowlisted",
            code="agent_unknown_tool",
            details={"tool_name": tool_name},
        )


class InvalidAgentToolArgumentsError(AgentError):
    def __init__(self, tool_name: str, errors: list[dict[str, Any]]) -> None:
        super().__init__(
            f"invalid arguments for agent tool '{tool_name}'",
            code="agent_invalid_tool_arguments",
            status_code=422,
            details={"tool_name": tool_name, "errors": errors},
        )


class AgentStepLimitError(AgentError):
    def __init__(self, limit: int) -> None:
        super().__init__(
            "agent tool-step limit exceeded",
            code="agent_step_limit",
            status_code=409,
            details={"max_tool_steps": limit},
        )


class AgentProviderUnavailableError(AgentError):
    def __init__(self) -> None:
        super().__init__(
            "configured LLM provider does not support agent tool calling",
            code="agent_provider_unavailable",
            status_code=503,
        )
