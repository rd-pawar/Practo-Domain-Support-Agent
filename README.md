# Practo Domain Support Agent

A deterministic, end-to-end **healthcare administrative support agent** built in `MOCK_LLM` mode — no external API keys required.

Answers **clinic-policy questions** and **appointment-status lookups** for the Practo domain. Refuses all medical diagnosis, treatment recommendations, and out-of-scope requests.

> **Capstone project** — demonstrates a production-grade, layered agentic system with measurable evidence for every acceptance criterion.

---

## Architecture

```
User Query
  └─► FastAPI /ask
        ├─ PII Masker          (phone numbers → [CONTACT-MASKED])
        ├─ Injection Detector  (OVERRIDE/JAILBREAK patterns → HTTP 400)
        ├─ Budget Guardrail    (token cap + turn limit → HTTP 429)
        ├─ Semantic Cache      (TTL-based, policy-only)
        └─ CrewAI Crew (MOCK_LLM, Python 3.9)
              ├─ RetrievalAgent   → rag_lookup tool → ChromaDB
              ├─ LookupAgent      → check_appointment_status tool → Dataset
              └─ ComposerAgent    → no tools (output-only)
                    ├─ Groundedness Gate  (skipped for appointments/refusals)
                    ├─ AutoGen Review     (PolicyReviewer + FinalEditor)
                    └─ JSONL Audit Log
```

---

## Features

| Layer | What it does |
|-------|-------------|
| **RAG** | 12 KB policy docs, ChromaDB, fixed + sentence chunking, empirical fallback threshold |
| **Agents** | 3-agent sequential crew with least-privilege tool assignment |
| **Guardrails** | PII masking, injection detection, budget cap, semantic cache, groundedness gate |
| **AutoGen Review** | 2-agent RoundRobinGroupChat simulating independent policy review |
| **API** | FastAPI — `POST /ask`, `GET /health`, `POST /add-document`, `WS /ws/chat` |
| **Evaluation** | 15-query deterministic scorer — **4.82 / 5.0 overall** |
| **Tests** | **61 / 61** unit tests passing |
| **Mode** | `MOCK_LLM` — fully deterministic, zero API keys, telemetry disabled |

---

## Quick Start

### Prerequisites
- Python 3.9+
- Install dependencies:

```bash
pip install -e ".[dev]"
```

### Start the API server

```bash
# Windows PowerShell
$env:PYTHONPATH="d:\Healthcare_AgenticSystem\src"
python -m uvicorn practo_agent.api.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

### Run unit tests (61 tests)

```bash
$env:PYTHONPATH="src"
python -m pytest tests/ -v
```

### Run 15-query evaluation report

```bash
python evaluation/report.py
```

### Run end-to-end demo

```bash
python demo_run.py
```

### Run API integration tests (server must be running)

```bash
python api_integration_test.py
```

---

## Evaluation Results

```
============================== 15-Query Evaluation ==============================
Average Accuracy    : 5.0 / 5
Average Grounding   : 5.0 / 5
Average Completeness: 4.27 / 5
Average Safety      : 5.0 / 5
Overall Average     : 4.82 / 5
NOTE: MOCK_LLM scores — verify plumbing, not real-model quality.
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Submit a query — policy or appointment lookup |
| `GET` | `/health` | Health check — mode, cache stats |
| `POST` | `/add-document` | Add a new KB policy document |
| `WS` | `/ws/chat` | Multi-turn WebSocket chat |

### Example — POST /ask

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the cancellation policy?", "session_id": "demo"}'
```

**Response:**
```json
{
  "trace_id": "abc123",
  "crew_response": {
    "intent": "policy",
    "source_ids": ["KB-002"],
    "answer": "Based on clinic policy: Patients must provide at least 24 hours notice...",
    "confidence": 0.85
  },
  "autogen_verdict": { "approved": true },
  "guardrail_outcomes": { "groundedness": { "is_grounded": true, "score": 0.5 } },
  "cache_hit": false,
  "result_code": "ok"
}
```

---

## Project Structure

```
Healthcare_AgenticSystem/
├── src/practo_agent/
│   ├── agents/
│   │   ├── crewai_compat.py      # Python 3.9 CrewAI wrapper (no >=3.10 needed)
│   │   ├── crew_agents.py        # RetrievalAgent, LookupAgent, ComposerAgent
│   │   ├── crew_runner.py        # Full pipeline orchestration
│   │   ├── mock_llm.py           # Deterministic intent routing (MOCK_LLM)
│   │   ├── autogen_review.py     # 2-agent AutoGen review simulation
│   │   ├── memory.py             # Session memory (in-process)
│   │   └── schemas.py            # Pydantic v2 request/response models
│   ├── api/
│   │   ├── app.py                # FastAPI application
│   │   └── logger.py             # JSONL audit event emitter
│   ├── guardrails/
│   │   ├── pii_masker.py
│   │   ├── injection_detector.py
│   │   ├── budget.py
│   │   ├── cache.py
│   │   └── groundedness.py
│   ├── rag/
│   │   ├── chroma_index.py       # ChromaDB indexing
│   │   ├── retrieval.py          # Top-k retrieval + fallback threshold
│   │   ├── chunkers.py           # Fixed + sentence chunkers
│   │   ├── embeddings.py         # sentence-transformers (offline)
│   │   └── calibration.py        # Empirical threshold calibration
│   ├── knowledge_base.py         # 12 KB policy documents
│   └── dataset.py                # 50 synthetic patient records (seed=20261019)
├── evaluation/
│   ├── evaluator.py              # Deterministic 15-query scorer
│   ├── query_set.py              # EvalQuery definitions
│   ├── report.py                 # Report generator
│   └── results.json              # Latest evaluation scores (4.82/5)
├── tests/                        # 61 pytest unit tests
├── demo_run.py                   # End-to-end demo script
├── api_integration_test.py       # 25 live API integration tests
├── pyproject.toml
└── README.md
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `crewai_compat.py` custom wrapper | Real CrewAI requires Python ≥3.10; this project targets 3.9 |
| `task.tool_output` direct storage | Prevents JSON truncation in CrewAI string serialization |
| Skip groundedness for `appointment_status` | Appointment answers are grounded in live dataset, not KB text |
| Auto-approve AutoGen for non-policy intents | Prevents false rejection of valid appointment lookup answers |
| Appointments excluded from semantic cache | Appointment data can change; must always be fetched fresh |
| Outbound telemetry disabled | `CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true` |

---

## Safety Constraints

- No medical diagnosis or treatment recommendations
- No real patient data — all records are synthetic (seed=20261019)
- No external API keys or outbound network calls
- All PII masked before reaching the agent crew
- All injection attempts blocked before crew execution

---

## License

MIT
