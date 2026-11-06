"""

app.py - FastAPI service for Practo Domain Support Agent.



Endpoints:

  POST /ask           - Main Q&A endpoint (policy + appointment)

  POST /add-document  - Controlled KB document addition

  GET  /ws/chat       - WebSocket multi-turn chat (disconnect-resilient)

  GET  /health        - Health check

"""



from __future__ import annotations



import json

import os

import time

import uuid

from typing import Any



import uvicorn

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse



# Disable telemetry before any CrewAI import

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

os.environ["OTEL_SDK_DISABLED"] = "true"



from practo_agent.agents.crew_runner import run as crew_run

from practo_agent.agents.memory import add_turn, get_context_string, reset_session

from practo_agent.agents.schemas import (

    AddDocRequest,

    AddDocResponse,

    AskRequest,

    AskResponse,

    AutoGenVerdict,

    CrewResponse,

)

from practo_agent.api.logger import emit_event

from practo_agent.guardrails.budget import check_budget

from practo_agent.guardrails.cache import get_cache

from practo_agent.guardrails.injection_detector import check_for_injection

from practo_agent.guardrails.pii_masker import mask_contact_numbers

from practo_agent.rag.chroma_index import (

    COLLECTION_FIXED,

    COLLECTION_SENTENCE,

    KB_VERSION,

    get_collection,

    load_kb_documents,

)

from practo_agent.rag.embeddings import embed_single



# ---------------------------------------------------------------------------

# App setup

# ---------------------------------------------------------------------------

app = FastAPI(

    title="Practo Domain Support Agent",

    description=(

        "Healthcare administrative support agent for clinic policy questions "

        "and appointment status inquiries. "

        "MOCK_LLM mode - no diagnosis, no treatment recommendations."

    ),

    version="1.0.0",

)



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"],

)



# Per-session turn counter (in-process only)

_session_turns: dict[str, int] = {}





# ---------------------------------------------------------------------------

# POST /ask

# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse, tags=["Core"])

async def ask(request: AskRequest) -> AskResponse:

    """

    Answer a clinic policy question or look up an appointment status.

    Runs the full CrewAI pipeline with guardrails and AutoGen review.

    """

    t_start = time.time()

    trace_id = str(uuid.uuid4())[:12]

    session_id = request.session_id

    turns_used = _session_turns.get(session_id, 0)



    masked_query = mask_contact_numbers(request.query)



    # Pre-check: injection

    injection = check_for_injection(masked_query)

    if injection.is_injection:

        emit_event(

            trace_id=trace_id,

            route="POST /ask",

            session_id=session_id,

            query=request.query,

            result_code="INJECTION_BLOCKED",

            guardrail_outcomes={"injection": injection.reason_code},

            duration_ms=round((time.time() - t_start) * 1000, 2),

        )

        raise HTTPException(status_code=400, detail=f"Request blocked: {injection.reason_code}")



    # Get session memory context

    memory_ctx = get_context_string(session_id)



    # Run crew pipeline

    result = crew_run(

        query=request.query,

        session_id=session_id,

        retrieval_strategy=request.retrieval_strategy,

        memory_context=memory_ctx,

        turns_used=turns_used,

        verbose=False,

    )



    duration_ms = round((time.time() - t_start) * 1000, 2)



    # Handle budget rejection

    if result["result_code"] not in ("ok", "CACHE_HIT"):

        emit_event(

            trace_id=trace_id,

            route="POST /ask",

            session_id=session_id,

            query=request.query,

            result_code=result["result_code"],

            guardrail_outcomes=result.get("guardrail_outcomes", {}),

            duration_ms=duration_ms,

            budget_tokens=result.get("budget_tokens", 0),

        )

        raise HTTPException(status_code=429, detail=result.get("error", "Budget exceeded"))



    # Build response objects

    crew_response = CrewResponse(**result["crew_response"])

    autogen_verdict = AutoGenVerdict(**result["autogen_verdict"])



    # Update session memory and turn counter

    add_turn(session_id, masked_query, crew_response.answer)

    _session_turns[session_id] = turns_used + 1



    # Log event

    emit_event(

        trace_id=trace_id,

        route="POST /ask",

        session_id=session_id,

        query=request.query,

        result_code="ok",

        intent=crew_response.intent,

        retrieval_ids=crew_response.source_ids,

        cache_state="hit" if result.get("cache_hit") else "miss",

        guardrail_outcomes=result.get("guardrail_outcomes", {}),

        duration_ms=duration_ms,

        budget_tokens=result.get("budget_tokens", 0),

    )



    return AskResponse(

        trace_id=trace_id,

        session_id=session_id,

        crew_response=crew_response,

        autogen_verdict=autogen_verdict,

        cache_hit=result.get("cache_hit", False),

        budget_used_tokens=result.get("budget_tokens", 0),
        result_code=result.get("result_code", "ok"),
        guardrail_outcomes=result.get("guardrail_outcomes", {}),
    )





# ---------------------------------------------------------------------------

# POST /add-document

# ---------------------------------------------------------------------------

@app.post("/add-document", response_model=AddDocResponse, tags=["Knowledge Base"])

async def add_document(request: AddDocRequest) -> AddDocResponse:

    """

    Add a new policy document to the knowledge base.

    Validates length and metadata, and indexes into both collections.

    """

    t_start = time.time()

    trace_id = str(uuid.uuid4())[:12]



    # Validate content length

    if len(request.content) < 50:

        raise HTTPException(status_code=400, detail="Document content too short (min 50 chars)")

    if len(request.content) > 5000:

        raise HTTPException(status_code=400, detail="Document content too long (max 5000 chars)")



    doc = request.model_dump()

    if not doc.get("parent_doc_id"):

        doc["parent_doc_id"] = doc["id"]



    # Index into both collections

    from practo_agent.rag.chunkers import chunk_document



    for strategy, collection_name in [("fixed", COLLECTION_FIXED), ("sentence", COLLECTION_SENTENCE)]:

        chunks = chunk_document(doc, strategy=strategy)

        if not chunks:

            continue

        collection = get_collection(collection_name)

        texts = [c.text for c in chunks]

        embeddings = embed_single  # placeholder; use batch below

        vecs = [embed_single(t) for t in texts]

        collection.upsert(

            ids=[c.chunk_id for c in chunks],

            embeddings=vecs,

            documents=texts,

            metadatas=[

                {

                    "parent_doc_id": c.parent_doc_id,

                    "chunking_strategy": c.chunking_strategy,

                    "kb_version": request.version,

                }

                for c in chunks

            ],

        )



    duration_ms = round((time.time() - t_start) * 1000, 2)

    emit_event(

        trace_id=trace_id,

        route="POST /add-document",

        session_id="system",

        query=f"add-document:{request.id}",

        result_code="ok",

        duration_ms=duration_ms,

    )



    return AddDocResponse(

        success=True,

        doc_id=request.id,

        message=f"Document '{request.id}' indexed into both collections (v{request.version}).",

    )





# ---------------------------------------------------------------------------

# WebSocket /ws/chat - multi-turn chat with disconnect handling

# ---------------------------------------------------------------------------

@app.websocket("/ws/chat")

async def websocket_chat(websocket: WebSocket) -> None:

    """

    WebSocket endpoint for multi-turn chat.

    Accepts: { "query": str, "session_id": str }

    Sends  : { "answer": str, "intent": str, "trace_id": str, ... }

    Catches WebSocketDisconnect and keeps the service alive.

    """

    await websocket.accept()

    client_session = str(uuid.uuid4())[:8]



    try:

        while True:

            raw = await websocket.receive_text()

            try:

                payload = json.loads(raw)

            except json.JSONDecodeError:

                await websocket.send_json({"error": "Invalid JSON payload"})

                continue



            query = payload.get("query", "").strip()

            session_id = payload.get("session_id", client_session)



            if not query:

                await websocket.send_json({"error": "Empty query"})

                continue



            t_start = time.time()

            trace_id = str(uuid.uuid4())[:12]



            # Guardrails (fast path)

            masked_query = mask_contact_numbers(query)

            injection = check_for_injection(masked_query)

            if injection.is_injection:

                await websocket.send_json({

                    "trace_id": trace_id,

                    "error": f"Blocked: {injection.reason_code}",

                })

                continue



            memory_ctx = get_context_string(session_id)

            result = crew_run(

                query=query,

                session_id=session_id,

                memory_context=memory_ctx,

                verbose=False,

            )



            duration_ms = round((time.time() - t_start) * 1000, 2)



            if result.get("crew_response"):

                crew_resp = result["crew_response"]

                add_turn(session_id, masked_query, crew_resp.get("answer", ""))

                response_payload = {

                    "trace_id": trace_id,

                    "session_id": session_id,

                    "answer": crew_resp.get("answer", ""),

                    "intent": crew_resp.get("intent"),

                    "source_ids": crew_resp.get("source_ids", []),

                    "cache_hit": result.get("cache_hit", False),

                    "duration_ms": duration_ms,

                }

            else:

                response_payload = {

                    "trace_id": trace_id,

                    "session_id": session_id,

                    "error": result.get("error", "Request rejected"),

                    "result_code": result.get("result_code"),

                }



            emit_event(

                trace_id=trace_id,

                route="WS /ws/chat",

                session_id=session_id,

                query=query,

                result_code=result.get("result_code", "ok"),

                intent=result.get("crew_response", {}).get("intent") if result.get("crew_response") else None,

                cache_state="hit" if result.get("cache_hit") else "miss",

                guardrail_outcomes=result.get("guardrail_outcomes", {}),

                duration_ms=duration_ms,

                budget_tokens=result.get("budget_tokens", 0),

            )



            await websocket.send_json(response_payload)



    except WebSocketDisconnect:

        # Client disconnected - service stays alive for future connections

        pass

    except Exception as e:

        # Catch-all: log and close gracefully

        try:

            await websocket.send_json({"error": f"Server error: {str(e)[:100]}"})

        except Exception:

            pass





# ---------------------------------------------------------------------------

# GET /health

# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])

async def health() -> dict:

    """Health check endpoint."""

    cache = get_cache()

    return {

        "status": "ok",

        "mode": "MOCK_LLM",

        "cache_stats": cache.stats(),

    }





# ---------------------------------------------------------------------------

# Run

# ---------------------------------------------------------------------------

if __name__ == "__main__":

    uvicorn.run("practo_agent.api.app:app", host="0.0.0.0", port=8000, reload=False)



# ---------------------------------------------------------------------------
# UI - Serve the React/HTML frontend from /ui
# ---------------------------------------------------------------------------
_UI_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ui')
)

if os.path.isdir(_UI_DIR):
    app.mount('/ui', StaticFiles(directory=_UI_DIR, html=True), name='ui')

@app.get('/', include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url='/ui/index.html')
