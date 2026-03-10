# Agentic AI for Service Ticket Reduction & Prevention

A production-style **Agentic AI system designed to reduce and prevent customer support service tickets** using machine learning, autonomous agents, and retrieval-augmented knowledge systems.

This project demonstrates how **multi-agent AI architectures** can triage support issues, resolve common problems automatically, and proactively reduce future ticket volumes.

The system combines:

- Machine Learning based ticket classification
- Retrieval-Augmented Generation (RAG)
- Multi-Agent routing and decision making
- Observability with Prometheus and Grafana
- ML lifecycle management with MLflow
- Containerized deployment using Docker

---

# Business Problem

Customer support teams often face **large volumes of service tickets**, many of which are repetitive and solvable with existing knowledge base information.

Common challenges include:

- Manual ticket triaging
- Long response times
- High operational costs
- Repetitive customer issues
- Lack of proactive issue prevention

In large organizations, support teams may process **thousands of tickets per day**, making automation essential.

This project demonstrates how **Agentic AI systems can reduce ticket volume by automatically resolving common issues and proactively guiding users toward solutions before tickets escalate.**

---

# Why Agentic AI?

Traditional ML systems simply classify tickets.

An **Agentic AI system goes further** by enabling autonomous decision-making.

In this system:

- agents analyze the ticket context
- determine the appropriate action
- retrieve knowledge if needed
- escalate complex cases

The result is a **self-directed system capable of both resolution and prevention.**

---

# System Architecture

```
Customer Ticket
      │
      ▼
   FastAPI Service
      │
      ▼
 Ticket Classifier (ML)
      │
      ▼
   Agent Router
 ┌───────────────┬───────────────┬───────────────┐
 │ FAQ Agent     │ Billing Agent │ Escalation    │
 │               │               │ Agent         │
 └───────────────┴───────────────┴───────────────┘
      │
      ▼
 RAG Knowledge Base (FAISS)
      │
      ▼
 Automated Response
      │
      ▼
 Monitoring (Prometheus + Grafana)
```

Key components:

| Component | Role |
|--------|------|
| FastAPI | API layer for predictions and agent responses |
| ML Models | Classify ticket categories |
| Agent Router | Determines which agent should handle the request |
| RAG Knowledge Base | Retrieves contextual answers |
| Monitoring Stack | Tracks system performance |

---

# Dataset

The system uses a **customer support dataset from HuggingFace**.

Dataset characteristics:

- Customer issue descriptions
- Ticket categories
- Answer column containing resolution text

The **answer column is used to construct the RAG knowledge base**, allowing agents to retrieve contextual responses rather than relying purely on classification.

Dataset usage:

| Purpose | Dataset Field |
|------|---------------|
| Model Training | Ticket text |
| Label Prediction | Ticket category |
| Knowledge Retrieval | Answer column |

---

# Setup Instructions

Clone the repository:

```bash
git clone https://github.com/amulyagupta1278/AgenticAi.git
cd AgenticAi
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Data Pipeline

The data pipeline prepares raw support tickets for training.

Pipeline steps:

1. Download dataset
2. Clean and preprocess text
3. Generate features for ML models

Pipeline flow:

```
download_data.py
      ↓
preprocess_data.py
      ↓
feature_engineering.py
```

This ensures that training data is standardized and suitable for model training.

---

# Model Training

The system evaluates multiple classifiers to determine the best performing model.

Models used:

- Logistic Regression
- Linear SVC
- Random Forest

Feature representation:

TF-IDF vectorization of ticket text.

Experiment tracking is handled using **MLflow**, allowing comparison of model performance across experiments.

Training workflow:

```
Data → Feature Extraction → Model Training → Evaluation → Model Selection
```

The best performing model is promoted for inference.

---

# RAG Knowledge Base

To provide contextual answers, the system builds a **retrieval-augmented knowledge base**.

The script:

```
build_rag_index.py
```

performs the following steps:

1. Generate embeddings from knowledge base answers
2. Store vectors in a FAISS index
3. Enable similarity-based retrieval

When agents require additional context, the RAG system retrieves relevant information from this knowledge base.

---

# Agent System

The system implements a **multi-agent architecture** where specialized agents handle different tasks.

Agents include:

### Classification Agent
Predicts ticket category using trained ML models.

### FAQ Agent
Handles frequently asked questions using the RAG knowledge base.

### Billing Agent
Handles billing-related issues and retrieves billing guidance.

### Escalation Agent
Routes complex or uncertain cases to human support.

### Response Agent
Formats the final response returned to the user.

Each agent has **specific decision thresholds and routing logic**, enabling the system to autonomously determine how to handle incoming tickets.

---

# API Reference

The system exposes REST endpoints through FastAPI.

### Predict Ticket Category

```
POST /predict
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text": "My payment failed but money was deducted"}'
```

Example response:

```
{
 "category": "billing_issue",
 "confidence": 0.93
}
```

---

### Agent Response Endpoint

```
POST /agent/respond
```

Returns the full agent-generated response including retrieved knowledge.

---

### Health Check

```
GET /health
```

Used for system monitoring and readiness checks.

---

# Monitoring and Observability

Production systems require visibility into performance.

The project integrates:

- **Prometheus** for metrics collection
- **Grafana** for visualization dashboards

Metrics tracked include:

| Metric | Description |
|------|-------------|
| api_requests_total | Total API requests |
| prediction_latency | Model inference time |
| agent_invocations | Number of agent decisions |
| ticket_resolution_rate | Automated resolution rate |

Monitoring stack is deployed using **docker-compose**.

---

# AgentOps Integration

AgentOps enables monitoring of agent behavior.

Capabilities include:

- tracking agent decisions
- measuring agent response latency
- debugging agent workflows

This helps ensure the system behaves reliably under real-world workloads.

---

# Simulation Results

A simulation was conducted using sample ticket workloads.

Results are stored in:

```
simulation_metrics.json
```

Example metrics:

| Metric | Result |
|------|--------|
| simulated_tickets | 1000 |
| automated_resolution_rate | 72% |
| escalation_rate | 18% |
| classification_accuracy | 91% |

These results demonstrate the system’s potential to **significantly reduce support ticket workload**.

---

# Testing

The project includes a comprehensive automated test suite.

Test coverage includes:

- API endpoints
- model inference
- agent routing
- data pipeline components

Test statistics:

```
62 automated tests
execution time: ~3.2 seconds
```

Tests can be executed with:

```bash
pytest
```

---

# Docker Deployment

The system can be deployed using Docker.

Build the image:

```bash
docker build -t agentic-ai .
```

Run using docker-compose:

```bash
docker-compose up
```

This launches:

- API server
- monitoring services
- supporting infrastructure

---

# Project Structure

```
AgenticAi/
├── src/
│   ├── data
│   ├── models
│   ├── agents
│   └── api
│
├── monitoring
│
├── tests
│
├── docker
│
├── Makefile
│
└── README.md
```

---

# Makefile Commands

Common project tasks are automated using Makefile commands.

| Command | Description |
|------|-------------|
| make setup | install dependencies |
| make download | download dataset |
| make ingest | run preprocessing |
| make train | train models |
| make api | start API server |
| make test | run tests |

---

# Configuration

System configuration is defined using:

```
config.yaml
```

Environment variables can also be used to control:

- model paths
- monitoring configuration
- API settings

---

# Real-World Impact

This system demonstrates how **Agentic AI architectures can improve support operations by:**

- reducing manual ticket triaging
- resolving common issues automatically
- retrieving contextual knowledge for faster support
- providing monitoring and operational visibility

By combining machine learning, autonomous agents, and observability infrastructure, the system shows how AI can **both reduce existing ticket loads and prevent future issues.**

---

# License

MIT License