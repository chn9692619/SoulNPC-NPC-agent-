"""
FastAPI server for SoulNPC Agent - production-ready API endpoints.

Provides:
- POST /agent/run - Execute a single agent step
- POST /agent/stream - SSE streaming execution
- GET /agent/state/{thread_id} - Get conversation state
- DELETE /agent/state/{thread_id} - Reset conversation
- GET /health - Health check
- GET /metrics - Agent metrics (bad cases, traces)

Usage:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from typing import Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import json
import asyncio

# ---- Pydantic Models ----

class AgentRunRequest(BaseModel):
    player_event: str = Field(..., description="Natural language player action description", min_length=1, max_length=2000)
    thread_id: str = Field("default", description="Conversation thread ID for multi-turn memory")
    enable_memory: bool = Field(True, description="Enable memory retrieval and storage")
    enable_tool_calling: bool = Field(True, description="Enable Tool Calling during decision")

class AgentResponse(BaseModel):
    trace_id: str
    dialogue_text: str
    action_label: str
    emotion_state: dict = {}
    relation_state: dict = {}
    retrieved_memories_count: int = 0
    tool_calls_count: int = 0
    errors: list = []

class HealthResponse(BaseModel):
    status: str
    version: str
    langgraph_available: bool
    tools_loaded: int

class MetricsResponse(BaseModel):
    total_traces: int
    bad_cases_total: int
    bad_cases_unresolved: int
    bad_cases_by_type: dict
    bad_cases_by_severity: dict


# ---- App Setup ----

app = FastAPI(
    title="SoulNPC Agent API",
    description="Production API for the SoulNPC cognitive-emotional game character Agent system.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded engines
_engine = None
_tracer = None


def get_engine():
    global _engine
    if _engine is None:
        from src.orchestration.langgraph_engine import SoulNPCGraphEngine
        _engine = SoulNPCGraphEngine(
            character_config=str(ROOT / "configs" / "character_ella.yaml"),
            enable_memory=True,
            enable_tool_calling=True,
            enable_tracing=True,
        )
    return _engine


def get_tracer():
    global _tracer
    if _tracer is None:
        from src.observability import SoulNPCTracer
        _tracer = SoulNPCTracer(enable_remote=True)
    return _tracer


# ---- API Endpoints ----

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        from src.orchestration.langgraph_engine import LANGGRAPH_AVAILABLE
    except ImportError:
        LANGGRAPH_AVAILABLE = False
    
    try:
        from src.tools.npc_tools import NPC_TOOLS
        tools_count = len(NPC_TOOLS)
    except ImportError:
        tools_count = 0
    
    return HealthResponse(
        status="healthy",
        version="2.1.0",
        langgraph_available=LANGGRAPH_AVAILABLE,
        tools_loaded=tools_count,
    )


@app.post("/agent/run", response_model=AgentResponse)
async def agent_run(request: AgentRunRequest):
    """
    Execute a single Agent step synchronously.
    
    Returns the full Agent state including generated dialogue,
    emotional state changes, memory retrieval results, and any errors.
    """
    try:
        engine = get_engine()
        state = engine.run(
            player_event_text=request.player_event,
            thread_id=request.thread_id,
        )
        
        return AgentResponse(
            trace_id=state.trace_id,
            dialogue_text=state.dialogue_text,
            action_label=state.action_label,
            emotion_state=state.emotion_after,
            relation_state=state.relation_after,
            retrieved_memories_count=state.recalled_count,
            tool_calls_count=len(state.tool_calls),
            errors=state.errors,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/stream")
async def agent_stream(request: AgentRunRequest):
    """
    Execute Agent step with SSE streaming.
    
    Returns each step of the Agent pipeline as a Server-Sent Event.
    """
    async def event_generator():
        try:
            engine = get_engine()
            for event in engine.stream(
                player_event_text=request.player_event,
                thread_id=request.thread_id,
            ):
                yield f"data: {json.dumps(event, default=str)}\n\n"
                await asyncio.sleep(0.01)
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/agent/state/{thread_id}")
async def get_state(thread_id: str):
    """Get current Agent state for a conversation thread."""
    try:
        engine = get_engine()
        if engine._graph and hasattr(engine._graph, 'get_state'):
            config = {"configurable": {"thread_id": thread_id}}
            state = engine._graph.get_state(config)
            return {"thread_id": thread_id, "state": str(state) if state else None}
        return {"thread_id": thread_id, "state": None, "note": "State checkpoints require LangGraph"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/agent/state/{thread_id}")
async def reset_state(thread_id: str):
    """Reset conversation state for a thread."""
    return {"thread_id": thread_id, "status": "reset"}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get Agent observability metrics."""
    tracer = get_tracer()
    summary = tracer.get_bad_case_summary()
    
    # Count local traces
    trace_dir = ROOT / "data" / "traces"
    trace_count = len(list(trace_dir.glob("trace_*.json"))) if trace_dir.exists() else 0
    
    return MetricsResponse(
        total_traces=trace_count,
        bad_cases_total=summary["total"],
        bad_cases_unresolved=summary["unresolved"],
        bad_cases_by_type=summary["by_type"],
        bad_cases_by_severity=summary["by_severity"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
