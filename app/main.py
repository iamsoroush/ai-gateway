"""FastAPI application entrypoint.

Wires together logging, the provider router (kept on ``app.state`` so it can be
swapped in tests), the API routes, and the exception handlers that render every
error in the consistent ``{"error": {...}}`` envelope.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.models.errors import ErrorBody, ErrorResponse, GatewayError
from app.services.router import ProviderRouter
from app.services.usage_store import InMemoryUsageStore
from app.utils.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger()

app = FastAPI(title="ai-gateway", version="0.1.0")

# Constructed once at startup. Tests may replace these.
app.state.provider_router = ProviderRouter(settings)
# In-memory MVP usage store; swap for a durable implementation later.
app.state.usage_store = InMemoryUsageStore()

app.include_router(router)


@app.exception_handler(GatewayError)
async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
    logger.warning(
        "gateway.error",
        extra={"context": {"code": exc.code, "type": exc.error_type, "status": exc.status_code}},
    )
    return JSONResponse(status_code=exc.status_code, content=exc.to_response().model_dump())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(
            message="Invalid request body.",
            type="invalid_request_error",
            code="invalid_request_body",
        )
    )
    # Surface the field location and reason for each error to aid debugging, but
    # deliberately drop Pydantic's "input"/"ctx" fields which would echo the
    # offending request values (and could contain prompt content / PHI).
    content = body.model_dump()
    content["error"]["details"] = [
        {"loc": list(err.get("loc", [])), "msg": err.get("msg"), "type": err.get("type")}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content=content)
