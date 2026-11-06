"""
crew_agents.py - Three CrewAI Agent definitions for Practo Domain Support.

Uses crewai_compat (Python 3.9-compatible implementation).

Agents:
  1. RetrievalAgent  - Policy retrieval from ChromaDB (rag_lookup tool)
  2. LookupAgent     - Appointment record lookup (check_appointment_status ONLY)
  3. ComposerAgent   - Compose final structured response (no direct tool access)

Least-autonomy enforcement:
  check_appointment_status is registered ONLY for LookupAgent.
"""

from __future__ import annotations

import json
import os
from typing import Any, Type

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from pydantic import BaseModel, Field

from practo_agent.agents.crewai_compat import Agent, BaseLLM, BaseTool
from practo_agent.agents.mock_llm import MockCrewLLM
from practo_agent.dataset import check_appointment_status
from practo_agent.rag.retrieval import retrieve


# ---------------------------------------------------------------------------
# Tool Input Schemas
# ---------------------------------------------------------------------------
class RAGLookupInput(BaseModel):
    query: str = Field(..., description="Policy question to search for.")
    strategy: str = Field("fixed", description="'fixed' or 'sentence'.")
    top_k: int = Field(3, description="Number of chunks to retrieve.")


class AppointmentLookupInput(BaseModel):
    appointment_id: str = Field(..., description="Appointment ID, e.g. APT-0001.")


# ---------------------------------------------------------------------------
# CrewAI-compatible Tools
# ---------------------------------------------------------------------------
class RAGLookupTool(BaseTool):
    """Retrieves relevant policy chunks from ChromaDB knowledge base."""
    name: str = "rag_lookup"
    description: str = (
        "Search the clinic policy knowledge base for information relevant to "
        "the user's question. Returns policy text chunks with source IDs and similarity scores."
    )

    def _run(self, query: str = "", strategy: str = "fixed", top_k: int = 3) -> str:
        try:
            resp = retrieve(query, strategy=strategy, top_k=top_k)
            if resp.fallback_triggered:
                return json.dumps({
                    "fallback_triggered": True,
                    "reason": resp.fallback_reason,
                    "results": [],
                })
            results = [
                {
                    "chunk_id": r.chunk_id,
                    "parent_doc_id": r.parent_doc_id,
                    "text": r.text,
                    "similarity": r.similarity,
                }
                for r in resp.results
            ]
            return json.dumps({
                "fallback_triggered": False,
                "collection": resp.collection_name,
                "results": results,
            })
        except Exception as e:
            return json.dumps({"error": str(e), "results": [], "fallback_triggered": True})


class AppointmentStatusTool(BaseTool):
    """Looks up appointment status by ID. ONLY for LookupAgent."""
    name: str = "check_appointment_status"
    description: str = (
        "Look up the status and details of an appointment by its ID (e.g. APT-0001). "
        "Returns status, category, escalation score. Never returns raw contact information."
    )

    def _run(self, appointment_id: str = "UNKNOWN") -> str:
        try:
            result = check_appointment_status(appointment_id.strip().upper())
            return json.dumps(result)
        except ValueError as e:
            return json.dumps({"found": False, "error": str(e)})
        except Exception as e:
            return json.dumps({"found": False, "error": f"Lookup failed: {e}"})


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------
_rag_tool = RAGLookupTool()
_appointment_tool = AppointmentStatusTool()
_mock_llm = MockCrewLLM()


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------
def make_retrieval_agent() -> Agent:
    """RetrievalAgent - only has rag_lookup tool (no appointment access)."""
    return Agent(
        role="Policy Retrieval Specialist",
        goal="Retrieve the most relevant clinic policy information from the knowledge base.",
        backstory=(
            "You are an expert at searching the Practo clinic's policy knowledge base. "
            "You retrieve policy documents and present only what is found in the database. "
            "You do not access patient records."
        ),
        tools=[_rag_tool],
        llm=_mock_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def make_lookup_agent() -> Agent:
    """LookupAgent - ONLY agent with check_appointment_status tool."""
    return Agent(
        role="Appointment Record Specialist",
        goal="Look up appointment records using the appointment ID provided by the user.",
        backstory=(
            "You are a specialist in retrieving appointment records. "
            "You only access records when a valid appointment ID is provided. "
            "You never invent or guess record details."
        ),
        tools=[_appointment_tool],  # ONLY this agent has this tool
        llm=_mock_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def make_composer_agent() -> Agent:
    """ComposerAgent - no direct tool access."""
    return Agent(
        role="Response Composer",
        goal="Compose a clear, accurate, and safe structured JSON response.",
        backstory=(
            "You are a skilled healthcare administrative communicator. "
            "You synthesize retrieved information into helpful responses. "
            "You never diagnose, recommend treatment, or respond beyond administrative scope."
        ),
        tools=[],   # No direct tool access
        llm=_mock_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
    )
