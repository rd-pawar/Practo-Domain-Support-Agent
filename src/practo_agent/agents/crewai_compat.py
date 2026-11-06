"""
crewai_compat.py - Lightweight Python 3.9-compatible CrewAI-compatible framework.

Provides Agent, Task, Crew, Process, and BaseTool classes that exactly match
the CrewAI interface used in the capstone, but run entirely on Python 3.9
with no external CrewAI package dependency.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BaseLLM (mirrors CrewAI's BaseLLM extension point)
# ---------------------------------------------------------------------------
class BaseLLM:
    """Base class for CrewAI-compatible LLM adapters."""
    model: str = "base"

    def call(self, messages: List[Dict[str, str]], tools: Optional[List] = None, **kwargs: Any) -> str:
        raise NotImplementedError

    def supports_function_calling(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 4096


# ---------------------------------------------------------------------------
# BaseTool (mirrors CrewAI's BaseTool)
# ---------------------------------------------------------------------------
class BaseTool(BaseModel):
    """Base class for CrewAI-compatible tools."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def run(self, **kwargs: Any) -> str:
        try:
            return self._run(**kwargs)
        except Exception as e:
            return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent:
    """CrewAI-compatible Agent."""

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[BaseLLM] = None,
        verbose: bool = False,
        allow_delegation: bool = False,
        max_iter: int = 3,
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools: Dict[str, BaseTool] = {t.name: t for t in (tools or [])}
        self.llm = llm
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.max_iter = max_iter
        # Store the raw tool output separately for easy extraction
        self.last_tool_output: Optional[str] = None
        self.last_tool_name: Optional[str] = None

    def execute_task(self, task: "Task") -> str:
        """Execute a task using available tools and the LLM."""
        if self.verbose:
            print(f"\n[{self.role}] Executing: {task.description[:80]}...")

        tool_results: Dict[str, Any] = {}
        self.last_tool_output = None
        self.last_tool_name = None

        # Always call all available tools (every agent only has the tools it should use)
        for tool_name, tool in self.tools.items():
            args = self._extract_tool_args(task.description, tool_name)
            if self.verbose:
                print(f"  [{self.role}] -> calling tool: {tool_name}")
            result = tool.run(**args)
            tool_results[tool_name] = result
            # Save last tool output for direct extraction by crew_runner
            self.last_tool_output = result
            self.last_tool_name = tool_name
            if self.verbose:
                print(f"  [{self.role}] <- {tool_name}: {result[:200]}")

        # Build context for LLM (include tool results)
        context = task.description
        if tool_results:
            context += "\n\nTool Results:\n" + json.dumps(tool_results, indent=2)

        # Call mock LLM
        if self.llm:
            messages = [{"role": "user", "content": context}]
            llm_result = self.llm.call(messages)
            # Store tool output on task for direct retrieval
            if self.last_tool_output:
                task.tool_output = self.last_tool_output
                task.tool_name = self.last_tool_name
            return llm_result

        # Fallback: return tool results as-is
        if tool_results:
            single = list(tool_results.values())
            result_str = single[0] if len(single) == 1 else json.dumps(tool_results)
            task.tool_output = result_str
            return result_str

        return json.dumps({
            "answer": task.expected_output or "Task completed.",
            "intent": "policy",
            "source_ids": [],
            "confidence": 0.5,
            "needs_escalation": False,
            "refusal_reason": None,
        })

    def _extract_tool_args(self, description: str, tool_name: str) -> Dict[str, Any]:
        """Extract tool arguments from task description using pattern matching."""
        import re

        if tool_name == "rag_lookup":
            # Pattern: 'for: "<query>"' or 'for: <query>\n'
            query = ""
            m = re.search(r'for:\s*"([^"]+)"', description)
            if m:
                query = m.group(1)
            else:
                m = re.search(r'for:\s*(.+?)[\n\r]', description)
                if m:
                    query = m.group(1).strip().strip('"')
                else:
                    # Take text between first pair of double-quotes (10+ chars)
                    m = re.search(r'"([^"]{10,})"', description)
                    if m:
                        query = m.group(1)
                    else:
                        query = description[:120]
            strategy = "sentence" if "sentence" in description else "fixed"
            return {"query": query, "strategy": strategy, "top_k": 3}

        elif tool_name == "check_appointment_status":
            m = re.search(r"\bAPT-\d{4}\b", description, re.IGNORECASE)
            apt_id = m.group(0).upper() if m else "UNKNOWN"
            return {"appointment_id": apt_id}

        return {}


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
class Task:
    """CrewAI-compatible Task."""

    def __init__(
        self,
        description: str,
        expected_output: str,
        agent: "Agent",
        context: Optional[List["Task"]] = None,
    ):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.context = context or []
        self.output: Optional[str] = None
        self.tool_output: Optional[str] = None  # Raw tool result (not truncated)
        self.tool_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------
class Process(Enum):
    sequential = "sequential"
    hierarchical = "hierarchical"


# ---------------------------------------------------------------------------
# CrewOutput (mirrors CrewAI's output object)
# ---------------------------------------------------------------------------
class CrewOutput:
    def __init__(self, raw: str):
        self.raw = raw

    def __str__(self) -> str:
        return self.raw


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------
class Crew:
    """CrewAI-compatible Crew orchestrator."""

    def __init__(
        self,
        agents: List[Agent],
        tasks: List[Task],
        process: Process = Process.sequential,
        verbose: bool = False,
    ):
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.verbose = verbose

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> CrewOutput:
        """Execute all tasks sequentially and return the last output."""
        outputs: List[str] = []

        for i, task in enumerate(self.tasks):
            output = task.agent.execute_task(task)
            task.output = output
            outputs.append(output)

            if self.verbose:
                print(f"\n[Crew] Task '{task.agent.role}' complete.")

        final_output = outputs[-1] if outputs else "{}"
        return CrewOutput(raw=final_output)
