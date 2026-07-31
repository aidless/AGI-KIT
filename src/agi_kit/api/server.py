"""FastAPI server factory + lifespan + middleware."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from agi_kit import __version__
from agi_kit.agents.factory import create_agent
from agi_kit.api.schemas import (
    ErrorResponse,
    HealthResponse,
    RunRequest,
    RunResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolSpec,
)
from agi_kit.config import get_settings
from agi_kit.llms.factory import create_llm
from agi_kit.observability import get_logger, metrics
from agi_kit.tools.base import ToolRegistry

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #

auth_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_auth(
    bearer: HTTPAuthorizationCredentials | None = Security(auth_scheme),
    api_key: str | None = Security(api_key_scheme),
) -> str:
    """Verify either JWT bearer or API key."""
    settings = get_settings()
    if not settings.api.enable_auth:
        return "anonymous"
    if api_key and api_key == settings.api.secret_key:
        return "apikey"
    if bearer:
        try:
            payload = jwt.decode(bearer.credentials, settings.api.secret_key, algorithms=["HS256"])
            return str(payload.get("sub", "user"))
        except jwt.PyJWTError as e:
            raise HTTPException(status_code=401, detail=f"invalid token: {e}")
    raise HTTPException(status_code=401, detail="missing credentials")


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("server_starting", version=__version__, env=settings.env)
    # Eagerly load LLM
    app.state.llm = create_llm(settings.llm)
    app.state.tools = ToolRegistry()
    log.info("server_ready", tools=len(app.state.tools), model=settings.llm.model)
    yield
    log.info("server_stopping")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AGI Kit API",
        version=__version__,
        description="Enterprise-grade AGI research platform API",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----- Routes -----

    @app.get("/healthz", response_model=HealthResponse)
    async def health() -> HealthResponse:
        s = get_settings()
        try:
            tools_count = len(ToolRegistry())
            llm_info = f"{s.llm.backend}:{s.llm.model}"
            return HealthResponse(
                status="ok",
                version=__version__,
                llm_backend=s.llm.backend,
                llm_model=s.llm.model,
                tools_count=tools_count,
            )
        except Exception as e:
            return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})

    @app.get("/metrics")
    async def prom_metrics():
        return JSONResponse(content=generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/tools", response_model=list[ToolSpec])
    async def list_tools(user: str = Depends(verify_auth)) -> list[ToolSpec]:
        reg = ToolRegistry()
        return [
            ToolSpec(name=t.name, description=t.description, parameters=t.parameters)
            for t in reg
        ]

    @app.post("/v1/tools/invoke", response_model=ToolCallResponse)
    async def invoke_tool(
        req: ToolCallRequest,
        user: str = Depends(verify_auth),
    ) -> ToolCallResponse:
        reg = ToolRegistry()
        try:
            tool = reg.get(req.tool)
            result = tool(**req.args)
            return ToolCallResponse(tool=req.tool, result=str(result))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/v1/agents/run", response_model=RunResponse)
    async def run_agent(
        req: RunRequest,
        user: str = Depends(verify_auth),
    ) -> RunResponse:
        s = get_settings()
        llm = app.state.llm
        if req.model:
            llm = create_llm(s.llm.model_copy(update={"model": req.model}))
        paradigm = req.paradigm or s.agent.paradigm
        agent = create_agent(
            paradigm=paradigm,
            llm=llm,
            tools=app.state.tools,
            max_steps=req.max_steps or s.agent.max_steps,
        )
        t0 = time.perf_counter()
        try:
            result = agent.run(req.task)
            dt = time.perf_counter() - t0
            return RunResponse(
                final=result,
                steps=len(agent.history) // 2,
                duration_s=round(dt, 3),
                model=llm.model,
                paradigm=paradigm,
                history_tail=[
                    m.to_dict() if hasattr(m, "to_dict") else {"role": m.role, "content": m.content}
                    for m in agent.history[-20:]
                ],
            )
        except Exception as e:
            log.exception("agent_run_failed", task=req.task[:80], error=str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.exception_handler(Exception)
    async def global_exc(request: Request, exc: Exception):
        log.exception("unhandled_error", path=str(request.url), error=str(exc))
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error=type(exc).__name__, detail=str(exc)).model_dump(),
        )

    return app


app = create_app()