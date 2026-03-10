# Agentic AI — Service Ticket Reduction & Prevention System

A multi-agent AI system that classifies, auto-resolves, routes, and prevents customer support tickets. Built on a traditional ML classifier (TF-IDF + sklearn), enhanced with a RAG pipeline (FAISS + sentence-transformers) for auto-resolution using real past support answers, and monitored with Prometheus + Grafana.

---

## The Business Problem

Support tickets cost **$15–25 each** on average. At 60K+ tickets/year, that's $900K–$1.5M in annual support costs. On top of that:

- **Misrouted tickets** add ~48 hours to resolution time
- **Repeat issues** (same root cause, different customers) waste agent time
- **No visibility** into trends means problems keep coming back

This system attacks the problem from three angles:

| Problem | Solution | Expected Impact |
|---------|----------|-----------------|
| High ticket volume | Auto-resolve common tickets via RAG (real past answers) | 30–40% ticket deflection |
| Slow routing | Smart category + priority routing with escalation detection | Fewer misroutes, faster resolution |
| Repeat issues | Trend analysis + proactive recommendations | Prevent tickets before they happen |

---

## Architecture

```
                         ┌──────────────────────────────┐
                         │     FastAPI  (/agent/resolve) │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                               ┌────────────────┐
                               │  Orchestrator   │
                               │    Agent        │
                               └───┬────┬────┬──┘
                                   │    │    │
                    ┌──────────────┘    │    └──────────────┐
                    ▼                   ▼                   ▼
           ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
           │  Classifier    │  │  Resolution     │  │  Routing       │
           │  Agent         │  │  Agent (RAG)    │  │  Agent         │
           │  (sklearn)     │  │  (FAISS+SBERT)  │  │  (rules)       │
           └────────────────┘  └────────────────┘  └────────────────┘
                                                          │
                                                   ┌──────┴──────┐
                                                   │  Prevention  │
                                                   │  Agent       │
                                                   └─────────────┘
```

**Pipeline flow:**
1. **Classify** — TF-IDF + sklearn model predicts ticket category + confidence
2. **Resolve** — If confidence is high enough, search FAISS index for similar past tickets. If similarity ≥ 0.40, return the real past answer
3. **Route** — If not auto-resolved, route to the right team with smart priority (escalation keywords, confidence-based bumps, ITIL ticket types)
4. **Prevent** — Trend analysis across categories, tag extraction, recommendations

---

## Dataset

**Source**: [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) on HuggingFace (61.8K tickets, CC-BY-NC-4.0)

Key columns used:

| Column | How It's Used |
|--------|---------------|
| `subject` + `body` | Combined input text for classification and RAG |
| `answer` | Real past support responses — powers the RAG knowledge base |
| `queue` (10 categories) | Classification target (which team handles it) |
| `type` | ITIL ticket type (Incident/Request/Problem/Change) — affects routing priority |
| `priority` | Enrichment for classifier + routing |
| `tag_1`..`tag_8` | Multi-label tags — used for trend analysis and keyword extraction |

The `answer` column is the key differentiator. Instead of a hand-crafted knowledge base, the RAG index is built from **actual past support resolutions** grouped by queue.

---

## Quick Start

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd AgenticAi
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download data from HuggingFace
make download

# 3. Preprocess and build features
make ingest
python -m src.features.feature_store

# 4. Train models (all 3 + auto-select best)
make train-all

# 5. Build RAG index from dataset answers
make build-rag

# 6. Generate supplementary keyword index
make generate-kb

# 7. Start the API
make api

# 8. Run simulation to see impact metrics
make simulate
```

---

## Agent System

### Classifier Agent (`src/agents/classifier_agent.py`)

Wraps the existing sklearn pipeline (TF-IDF + LogisticRegression/SVC/RF). Adds confidence tier logic:

- **High** (≥ 0.75): Resolution agent can auto-resolve
- **Medium** (≥ 0.45): Resolution attempts, routing bumps priority one level
- **Low** (< 0.45): Skip resolution, escalate/route with high priority

### Resolution Agent (`src/agents/resolution_agent.py`)

RAG-based auto-resolution using FAISS + sentence-transformers:

1. Embed the incoming ticket with `all-MiniLM-L6-v2` (384-dim)
2. Search the FAISS index (IndexFlatIP — cosine similarity on normalized vectors)
3. Filter results to same predicted category
4. If best match similarity ≥ 0.40, return the real past answer
5. Falls back gracefully if FAISS/sentence-transformers aren't installed

The FAISS index is pre-built by `scripts/build_rag_index.py` from the dataset's `answer` column.

### Routing Agent (`src/agents/routing_agent.py`)

Smart routing with escalation detection:

- **Category → team mapping** from `knowledge_base/routing_rules.json`
- **Escalation keywords**: "outage", "down", "security", "breach", "fraud", etc.
- **Priority logic**:
  - Escalation keyword found → critical
  - Incident type + low confidence → critical (possibly misrouted emergency)
  - Low confidence → high (needs human triage)
  - Medium confidence → bump one level
  - High confidence → category default

### Prevention Agent (`src/agents/prevention_agent.py`)

Analyzes historical data for trends and generates proactive recommendations:

- Category distribution and trend direction (increasing/decreasing/stable)
- Top tags per category (from tag_1..tag_8 columns)
- Per-category and overall prevention recommendations
- Knowledge gap analysis (categories where auto-resolution is weak)

### Orchestrator (`src/agents/orchestrator.py`)

Coordinates the full pipeline: classify → resolve → route. Each step is independently timed for latency analysis. Integrates with AgentOps for session tracking.

---

## API Endpoints

### Original Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Classify a ticket (sklearn only) |
| `GET` | `/health` | Model health check |
| `GET` | `/classes` | List known categories |

### Agent Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/agent/resolve` | Full pipeline: classify → resolve → route |
| `GET` | `/agent/insights` | Prevention trend analysis + recommendations |
| `GET` | `/agent/status` | All agent health checks |

### Monitoring

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/metrics` | Prometheus metrics (request counts, latency, agent outcomes) |

### Example: Resolve a Ticket

```bash
curl -X POST http://localhost:8000/agent/resolve \
  -H "Content-Type: application/json" \
  -d '{"description": "My app keeps crashing when I try to upload files", "ticket_id": "T-1234"}'
```

Response:
```json
{
  "ticket_id": "T-1234",
  "status": "auto_resolved",
  "classification": {
    "category": "technical_support",
    "confidence": 0.87,
    "confidence_tier": "high"
  },
  "resolution": {
    "resolved": true,
    "answer": "Please try clearing your browser cache and restarting the application...",
    "similarity_score": 0.82
  },
  "routing": null,
  "processing_time_ms": 42.5
}
```

---

## Monitoring (Prometheus + Grafana)

The API exposes Prometheus metrics at `/metrics`:

- `http_requests_total` — request count by method/endpoint/status
- `http_request_duration_seconds` — latency histogram
- `predictions_total` — predictions by category
- `prediction_confidence` — confidence score distribution
- `agent_resolutions_total` — outcomes (auto_resolved / routed / escalated)
- `agent_pipeline_duration_seconds` — full pipeline latency

### Start Monitoring Stack

```bash
# Start app + Prometheus + Grafana
cd deployment
docker compose --profile monitoring up -d
```

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

A pre-built dashboard with 7 panels is auto-provisioned in Grafana:
request rate, latency percentiles, category breakdown, confidence distribution, agent outcomes, error rate, and pipeline latency.

---

## Simulation

Run tickets through the full agent pipeline to measure impact:

```bash
python scripts/run_simulation.py --sample 500
```

This outputs:
- Classification accuracy
- Auto-resolution rate and accuracy
- Routing and escalation stats
- Average latency per ticket
- Estimated annual cost savings (based on $20/ticket, extrapolated to 60K tickets/year)
- Per-category breakdown

Results are saved to `data/processed/simulation_results.csv` and `data/processed/simulation_metrics.json`.

---

## AgentOps Integration

The system integrates with [AgentOps](https://agentops.ai) for agent session tracking and observability. Set your API key to enable it:

```bash
export AGENTOPS_API_KEY="your-key-here"
```

Each ticket processed through `/agent/resolve` creates a trace in AgentOps with timing for every agent step. Works as a no-op if the key isn't set.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src

# Just agent tests
pytest tests/test_agents.py -v
pytest tests/test_orchestrator.py -v
pytest tests/test_agent_api.py -v
```

---

## Docker Deployment

Train the model and build the RAG index before building the Docker image:

```bash
# Build image (copies models/ and knowledge_base/ into the image)
docker build -t ticket-classifier:latest .

# Run standalone
docker run --rm -p 8000:8000 ticket-classifier:latest

# Full stack with docker compose
cd deployment
docker compose up                          # app only
docker compose --profile monitoring up     # app + prometheus + grafana
docker compose --profile mlflow up         # app + mlflow UI
```

---

## Project Structure

```
AgenticAi/
├── src/
│   ├── agents/                    # Multi-agent system
│   │   ├── base.py                # Abstract base agent
│   │   ├── schemas.py             # Pydantic models
│   │   ├── classifier_agent.py    # TF-IDF + sklearn classifier
│   │   ├── resolution_agent.py    # RAG auto-resolution (FAISS + SBERT)
│   │   ├── routing_agent.py       # Smart routing + escalation
│   │   ├── prevention_agent.py    # Trend analysis + recommendations
│   │   └── orchestrator.py        # Pipeline coordinator
│   ├── api/
│   │   ├── main.py                # FastAPI application
│   │   ├── agent_router.py        # /agent/* endpoints
│   │   ├── metrics.py             # Prometheus instrumentation
│   │   └── schemas.py             # API request/response models
│   ├── data/
│   │   ├── download_data.py       # HuggingFace dataset downloader
│   │   ├── preprocess.py          # Text cleaning + label normalization
│   │   └── build_training_dataset.py
│   ├── features/
│   │   ├── build_features.py      # TF-IDF vectorizer + train/test split
│   │   └── feature_store.py       # Cached feature matrices
│   ├── models/
│   │   ├── train.py               # Model training (3 classifiers + MLflow)
│   │   └── evaluate.py            # Evaluation metrics
│   └── agentops_config.py         # AgentOps SDK setup
├── scripts/
│   ├── build_rag_index.py         # Build FAISS index from dataset answers
│   ├── generate_kb.py             # Extract TF-IDF keywords per category
│   └── run_simulation.py          # Run tickets through pipeline, measure impact
├── knowledge_base/
│   └── routing_rules.json         # Category → team mapping + escalation keywords
├── monitoring/
│   ├── prometheus.yml             # Prometheus scrape config
│   └── grafana/                   # Grafana provisioning + dashboard JSON
├── deployment/
│   └── docker-compose.yml         # App + Prometheus + Grafana + MLflow
├── tests/
│   ├── test_api.py                # Original API tests
│   ├── test_pipeline.py           # Data pipeline tests
│   ├── test_agents.py             # Agent unit tests
│   ├── test_orchestrator.py       # Integration tests
│   └── test_agent_api.py          # Agent API endpoint tests
├── notebooks/
│   └── 01_eda.ipynb               # Exploratory data analysis
├── config.yaml
├── Dockerfile
├── Makefile
├── requirements.txt
└── run_pipeline.py
```

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make setup` | Install dependencies |
| `make download` | Download dataset from HuggingFace |
| `make ingest` | Preprocess raw data |
| `make train` | Train default model (logistic regression) |
| `make train-all` | Train all 3 models, auto-select best |
| `make build-rag` | Build FAISS index from dataset answers |
| `make generate-kb` | Generate TF-IDF keyword index |
| `make simulate` | Run 500-ticket simulation with impact metrics |
| `make api` | Start FastAPI dev server |
| `make test` | Run all tests |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run container locally |
| `make monitoring` | Start Prometheus + Grafana stack |
| `make mlflow-ui` | Start MLflow tracking UI |
| `make clean` | Remove cached files |
