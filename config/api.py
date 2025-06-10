"""API configuration for MyJobSpyAI."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import settings

logger = logging.getLogger('api')


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MyJobsSpyAI API",
        description="API for MyJobsSpyAI job search and analysis platform",
        version=settings.settings.get('version', '1.0.0'),
        debug=settings.settings.debug,
        docs_url="/docs" if settings.settings.debug else None,
        redoc_url="/redoc" if settings.settings.debug else None,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add middleware for logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = datetime.utcnow()

        # Log request
        request_body = await request.body()
        request_headers = dict(request.headers)

        # Hide sensitive headers
        for header in ['authorization', 'cookie', 'set-cookie']:
            if header in request_headers:
                request_headers[header] = '***REDACTED***'

        logger.info(
            "Request received",
            extra={
                "request": {
                    "method": request.method,
                    "url": str(request.url),
                    "headers": request_headers,
                    "query_params": dict(request.query_params),
                    "path_params": request.path_params,
                    "client": (
                        f"{request.client.host}:{request.client.port}"
                        if request.client
                        else None
                    ),
                    "body": request_body.decode() if request_body else None,
                }
            },
        )

        # Process request
        try:
            response = await call_next(request)
            process_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Log response
            logger.info(
                "Request completed",
                extra={
                    "status_code": response.status_code,
                    "process_time_ms": process_time,
                    "response_headers": dict(response.headers),
                },
            )

            return response

        except Exception as e:
            logger.error(f"Request failed: {str(e)}", exc_info=True)
            raise

    # Add exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.warning(
            "HTTP exception",
            extra={
                "status_code": exc.status_code,
                "detail": str(exc.detail),
                "path": request.url.path,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc.detail)},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        logger.warning(
            "Validation error",
            extra={"errors": exc.errors(), "body": exc.body, "path": request.url.path},
        )
        return JSONResponse(
            status_code=422,
            content={"detail": "Validation Error", "errors": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception", exc_info=True, extra={"path": request.url.path}
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
        )

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "ok",
            "environment": settings.settings.environment,
            "debug": settings.settings.debug,
            "version": "1.0.0",
        }

    return app
