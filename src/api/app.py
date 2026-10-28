"""
src/api/app.py
==============
FastAPI application factory for LeafMind backend.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.config import load_api_config
from src.api.routes import router
from src.orchestrator.nodes import sanitize_error

logger = logging.getLogger("leafmind.api")


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create and configure FastAPI application instance."""
    cfg = load_api_config(config_path)

    app = FastAPI(
        title="LeafMind Diagnostic Advisory API",
        description="Multi-agent agricultural disease diagnostic and risk advisory backend.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. CORS Configuration
    # Read ALLOWED_ORIGINS env var or fall back to config
    env_origins = os.getenv("ALLOWED_ORIGINS")
    if env_origins:
        origins = [o.strip() for o in env_origins.split(",") if o.strip()]
    else:
        origins = cfg.get("cors", {}).get("allowed_origins", [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://localhost:3000",
            "http://localhost:5173",
        ])

    # Guard against wildcard when credentials are true
    if "*" in origins:
        origins = [o for o in origins if o != "*"]
        if not origins:
            origins = ["http://localhost:8501"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. Global Exception Handler for Unhandled Errors (No traceback / path leak)
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        sanitized = sanitize_error(exc)
        logger.error("Unhandled API exception on %s: %s", request.url.path, sanitized)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Internal server error: {sanitized}"},
        )

    # 3. Mount Routes
    app.include_router(router)

    return app


# Module-level default application instance for uvicorn
app = create_app()
