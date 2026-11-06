"""
schemas.py - Pydantic models for Practo Domain Support Agent.

All schemas use Pydantic v2 field validators.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# CrewAI Agent Response
# ---------------------------------------------------------------------------
class CrewResponse(BaseModel):
    """Structured response from the CrewAI Composer agent."""
    answer: str = Field(..., min_length=1, description="Final answer text.")
    intent: Literal["policy", "appointment_status", "mixed", "out_of_scope"] = Field(
        ..., description="Detected and confirmed intent category."
    )
    source_ids: List[str] = Field(default_factory=list, description="KB or APT IDs cited.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Answer confidence [0-1].")
    needs_escalation: bool = Field(..., description="True if urgent attention required.")
    refusal_reason: Optional[str] = Field(None, description="Reason if request was refused.")

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)


# ---------------------------------------------------------------------------
# AutoGen Review Verdict
# ---------------------------------------------------------------------------
class AutoGenVerdict(BaseModel):
    """Verdict from the AutoGen two-agent review."""
    approved: bool = Field(..., description="True if answer is approved without changes.")
    final_answer: str = Field(..., description="Final answer (may be revised).")
    reason: str = Field(..., description="Reason for approval or revision.")


# ---------------------------------------------------------------------------
# FastAPI Request / Response Models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    """POST /ask request payload."""
    query: str = Field(..., min_length=1, max_length=2000, description="User's question.")
    session_id: str = Field(default="default", description="Session identifier for memory.")
    retrieval_strategy: Literal["fixed", "sentence"] = Field(
        default="fixed", description="Chunking strategy for retrieval."
    )


class AskResponse(BaseModel):
    """POST /ask response payload."""
    trace_id: str
    session_id: str
    crew_response: CrewResponse
    autogen_verdict: AutoGenVerdict
    cache_hit: bool
    budget_used_tokens: int
    result_code: str = Field(default="ok")
    guardrail_outcomes: dict = Field(default_factory=dict)


class AddDocRequest(BaseModel):
    """POST /add-document request payload."""
    id: str = Field(..., description="Unique document ID (e.g. KB-013).")
    topic: str = Field(..., description="Topic name.")
    version: str = Field(default="1.0", description="Document version.")
    effective_date: str = Field(default="2024-01-01", description="Effective date YYYY-MM-DD.")
    parent_doc_id: Optional[str] = Field(None, description="Parent document ID (defaults to id).")
    content: str = Field(..., min_length=50, max_length=5000, description="Policy content.")


class AddDocResponse(BaseModel):
    """POST /add-document response payload."""
    success: bool
    doc_id: str
    message: str


# ---------------------------------------------------------------------------
# Structured JSONL Log Event
# ---------------------------------------------------------------------------
class LogEvent(BaseModel):
    """One JSONL audit log event."""
    trace_id: str
    timestamp: str
    route: str
    session_id: str
    query_masked: str
    intent: Optional[str]
    retrieval_ids: List[str]
    cache_state: str
    guardrail_outcomes: dict
    result_code: str
    duration_ms: float
    budget_tokens: int
