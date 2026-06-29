"""Gateway error types and the consistent JSON error envelope.

All errors returned to callers share the shape::

    {"error": {"message": "...", "type": "invalid_request_error", "code": "..."}}

Each :class:`GatewayError` subclass carries the HTTP status, ``type`` and
``code`` so the FastAPI exception handler can render it uniformly.
"""

from __future__ import annotations

from pydantic import BaseModel


class ErrorBody(BaseModel):
    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class GatewayError(Exception):
    """Base class for errors that map to a consistent JSON response."""

    status_code: int = 500
    error_type: str = "internal_error"
    code: str = "internal_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorBody(message=self.message, type=self.error_type, code=self.code)
        )


class UnknownModelError(GatewayError):
    """The requested model alias is not in the registry."""

    status_code = 404
    error_type = "invalid_request_error"
    code = "model_not_found"


class UnsupportedProviderError(GatewayError):
    """A model alias points at a provider the gateway has no adapter for."""

    status_code = 500
    error_type = "internal_error"
    code = "unsupported_provider"


class UnsupportedContentError(GatewayError):
    """The selected provider/model cannot handle one of the content types sent."""

    status_code = 400
    error_type = "invalid_request_error"
    code = "unsupported_content_type"


class UnsupportedFeatureError(GatewayError):
    """The selected provider/model cannot handle a requested API feature."""

    status_code = 400
    error_type = "invalid_request_error"
    code = "unsupported_feature"


class MissingAPIKeyError(GatewayError):
    """The provider's API key is not configured."""

    status_code = 500
    error_type = "internal_error"
    code = "missing_api_key"


class ProviderRequestError(GatewayError):
    """The upstream provider call failed."""

    status_code = 502
    error_type = "provider_error"
    code = "provider_request_failed"
