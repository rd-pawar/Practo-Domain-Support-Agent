"""
conftest.py – Shared pytest fixtures.

Sets required environment variables before any test import to ensure
MOCK_LLM mode and telemetry-disabled state.
"""

import os
import sys
from pathlib import Path

# Must be before any CrewAI import
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
