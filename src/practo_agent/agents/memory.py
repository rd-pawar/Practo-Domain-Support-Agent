"""
memory.py - Session memory using LangChain InMemoryChatMessageHistory.

Keyed by session_id (in-process only; not persisted across restarts).
Demonstrates multi-turn context and fresh-session isolation.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# In-process session store
# ---------------------------------------------------------------------------
_SESSION_STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return (or create) the InMemoryChatMessageHistory for *session_id*."""
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


def add_turn(session_id: str, human: str, ai: str) -> None:
    """Append a human+AI message pair to the session history."""
    hist = get_history(session_id)
    hist.add_message(HumanMessage(content=human))
    hist.add_message(AIMessage(content=ai))


def get_context_string(session_id: str, max_turns: int = 5) -> str:
    """
    Return recent session history as a plain text string for use in prompts.

    Parameters
    ----------
    session_id : Session identifier.
    max_turns  : Maximum number of human+AI exchanges to include.
    """
    hist = get_history(session_id)
    messages = hist.messages[-(max_turns * 2):]   # last N exchanges
    lines: list[str] = []
    for msg in messages:
        role = "Patient" if isinstance(msg, HumanMessage) else "Agent"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def reset_session(session_id: str) -> None:
    """Clear the history for a session (simulates fresh session)."""
    _SESSION_STORE[session_id] = InMemoryChatMessageHistory()


def list_sessions() -> list[str]:
    """Return all active session IDs."""
    return list(_SESSION_STORE.keys())


# ---------------------------------------------------------------------------
# Demo: same-session vs fresh-session
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Session Memory - Demo")
    print("=" * 60)

    SID_A = "session-demo-A"
    SID_B = "session-demo-B"

    # Session A - multi-turn conversation
    add_turn(SID_A, "How far ahead should I book an appointment?",
             "You can book up to 30 days in advance via the clinic portal.")
    add_turn(SID_A, "What about Cardiology?",
             "For Cardiology, booking 3 to 5 business days ahead is recommended.")

    print(f"\n[Session {SID_A}] Context (2 turns):")
    print(get_context_string(SID_A))

    # Session B - fresh, no prior facts
    print(f"\n[Session {SID_B}] Context (fresh, no history):")
    fresh_ctx = get_context_string(SID_B)
    print(repr(fresh_ctx) if fresh_ctx else "  (empty - no prior turns)")

    # Reset A and verify it's empty
    reset_session(SID_A)
    print(f"\n[Session {SID_A}] After reset:")
    print(repr(get_context_string(SID_A)) or "  (empty)")

    print("\n[memory] Active sessions:", list_sessions())
