# Sigma — Horo Knowledge Co-Pilot

> A private, hallucination-resistant RAG system for founders. Upload your pitch decks, loan policies, handbooks, and finance sheets — then ask questions in plain English. Horo answers only from your documents, cites every source, and tells you exactly what's missing.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Evaluation & Anti-Hallucination](#evaluation--anti-hallucination)
- [Security & Tenant Isolation](#security--tenant-isolation)
- [Known Limitations](#known-limitations)

---

## Overview

Sigma is a multi-tenant RAG (Retrieval-Augmented Generation) platform. Each founder gets a private knowledge base scoped to their tenant. Horo — the AI co-pilot — retrieves the most relevant document chunks, streams a grounded answer, and cites the exact document and page number it used.

If the answer isn't in your documents, Horo says so — and either tells you which file to upload, or shows an industry benchmark so you're never left empty-handed.

**Example interactions:**

| Question | Behaviour |
|---|---|
| "What's the maximum loan size for first-time borrowers?" | Answer + `Loan Policy, pp.2, 7` |
| "List the onboarding steps for our program." | Structured list + `Handbook, pp.3, 4, 5` |
| "What's our CAC?" *(no finance doc uploaded)* | Refusal + upload prompt + SaaS CAC benchmark card |

---

## Features

- **Private knowledge base** — files and chunks are scoped strictly to your tenant ID; zero cross-tenant access enforced at SQL level
- **Streaming answers** — tokens stream to the UI in real time via SSE
- **Source citations** — every answer shows the document name and pages retrieved, grouped per document
- **Grounded-only responses** — similarity gate blocks answers when no relevant chunk is found; the LLM is instructed never to use prior knowledge
- **Contextual refusals** — when Horo can't answer, it identifies *which type* of document is missing (finance sheet, loan policy, handbook, etc.)
- **Industry benchmarks** — refusals for known metrics (CAC, LTV, burn rate, runway, loan size, etc.) show a benchmark card so founders get value even without the document
- **Hallucination scoring** — every answer is scored post-generation for faithfulness (LLM-based) and grounding (token-level)
- **Citation mismatch detection** — inline `p.X` citations in the answer are cross-validated against retrieved source metadata; mismatches are flagged in the UI and eval log
- **PII masking** — SSNs, emails, card numbers, and long numeric IDs are redacted before embedding and storage
- **Conversation history** — prior turns are injected into context with a token budget so retrieved chunks are never crowded out
- **Eval dashboard** — live per-query table showing confidence, faithfulness, grounding score, source count, and answered/refused status
- **Auto-generated prompt suggestions** — 3 starter questions are generated per uploaded document and surfaced in the UI

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Frontend                         │
│   React (Vite) · SSE streaming · Eval dashboard         │
└────────────────────────┬────────────────────────────────┘
                         │ REST + SSE
┌────────────────────────▼────────────────────────────────┐
│                     FastAPI Backend                     │
│                                                         │
│  /upload   →  Extract → Chunk → Embed → Store           │
│  /chat     →  Embed query → Retrieve → Stream → Score   │
│  /conversations  →  History CRUD                        │
│  /files    →  Document management                       │
│  /prompts  →  Suggested questions                       │
└──────┬─────────────────────────┬───────────────────────┘
       │                         │
┌──────▼──────┐         ┌────────▼────────┐
│  PostgreSQL  │         │  Azure OpenAI   │
│  + pgvector  │         │  Embeddings +   │
│  (chunks,    │         │  Chat (GPT-4)   │
│  documents,  │         └─────────────────┘
│  messages)   │
└─────────────┘
```

### Request flow — `/chat`

```
User question
    │
    ▼
Greeting detection (regex + set lookup, no API call)
    │
    ├── greeting? ──► Friendly reply, skip pipeline entirely
    │
    ▼
Embed question (Azure OpenAI text-embedding-ada-002)
    │
    ▼
pgvector cosine search  ←── WHERE tenant_id = $X  (hard isolation)
    │
    ├── no chunks? ──► "Knowledge base is empty" + upload prompt
    │
    ▼
Similarity gate  ──── below threshold? ──► Refusal + contextual hint + benchmark card
    │
    ▼
Stream answer (GPT-4, temperature=0, context-only instructions)
    │
    ├──► Token-level grounding score  (synchronous, no API call)
    └──► LLM faithfulness score       (second GPT-4 call, post-stream)
    │
    ▼
SSE  event: done  →  sources, confidence, faithfulness, grounding, unsupported_claims
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, plain CSS-in-JS |
| Backend | Python 3.11, FastAPI, asyncpg |
| Database | PostgreSQL 15 + pgvector extension |
| Embeddings | Azure OpenAI `text-embedding-ada-002` |
| LLM | Azure OpenAI GPT-4 (streaming) |
| Document parsing | pdfplumber, python-docx, openpyxl |
| Deployment | Docker / any ASGI host (e.g. Railway, Render) |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node 18+
- PostgreSQL 15 with the `pgvector` extension enabled
- An Azure OpenAI resource with a chat deployment and an embeddings deployment

### 1. Clone the repo

```bash
git clone https://github.com/your-org/sigma-horo.git
cd sigma-horo
```

### 2. Backend setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create the database schema:

```bash
psql $DATABASE_URL -f schema.sql
```

Copy and fill in environment variables:

```bash
cp .env.example .env
```

Start the API server:

```bash
uvicorn main:app --reload --port 8000
```

### 3. Frontend setup

```bash
cd frontend
npm install
cp .env.example .env        # set VITE_API_URL and VITE_TENANT_ID
npm run dev
```

Open `http://localhost:5173`.

---

## Environment Variables

### Backend (`.env`)

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | e.g. `2024-02-01` |
| `AZURE_CHAT_DEPLOYMENT` | Name of your GPT-4 deployment |
| `AZURE_EMBED_DEPLOYMENT` | Name of your embeddings deployment |
| `SIM_THRESHOLD` | Cosine similarity gate (default `0.45`) |
| `TOP_K` | Chunks retrieved per query (default `6`) |
| `CHUNK_SIZE` | Words per text chunk (default `800`) |
| `CHUNK_OVERLAP` | Overlap between chunks (default `150`) |

### Frontend (`.env`)

| Variable | Description |
|---|---|
| `VITE_API_URL` | Backend base URL, e.g. `http://localhost:8000` |
| `VITE_TENANT_ID` | Tenant identifier for this session |

---

## Project Structure

```
sigma-horo/
├── backend/
│   ├── main.py                  # FastAPI app, router registration
│   ├── core/
│   │   ├── config.py            # Settings (pydantic-settings)
│   │   └── database.py          # asyncpg pool
│   ├── routers/
│   │   ├── routers.py           # Shared deps (get_tenant)
│   │   ├── upload.py            # File ingestion pipeline
│   │   ├── chat.py              # Streaming chat endpoint
│   │   ├── files.py             # Document management
│   │   ├── prompts.py           # Suggested questions
│   │   └── conversations.py     # Conversation + message CRUD
│   ├── services/
│   │   ├── services.py          # Chunker, embedder, retriever, generator, scorers
│   │   └── extractor.py         # PDF / DOCX / XLSX / TXT parsing
│   └── schema.sql               # DB schema (conversations, messages, chunks, documents)
└── frontend/
    └── src/
        └── App.jsx              # Full React UI — chat, knowledge base, eval tab
```

---

## How It Works

### Document ingestion (`/upload`)

1. File is hashed (SHA-256) — duplicates are rejected immediately
2. Text and tables are extracted per page using `pdfplumber` / `python-docx` / `openpyxl`
3. Tables are serialised to Markdown so the LLM reads exact cell values
4. Prose is split into overlapping word-window chunks; tables are kept intact as single chunks
5. PII is masked in all chunk content before embedding or storage
6. Chunks are embedded in batches of 100 and stored as `vector(1536)` in pgvector
7. A tag is inferred from the filename (`Policy`, `Finance`, `Operations`, `Pitch`, `Document`)
8. 3 starter questions are generated via `asyncio.create_task` — fire-and-forget so the upload response is not blocked

### Retrieval & answer generation (`/chat`)

1. Greeting / small-talk detection runs first — matched inputs return a friendly reply immediately with no API call
2. The user's question is embedded
3. pgvector performs a cosine similarity search scoped to `tenant_id` — hard `WHERE` clause prevents any cross-tenant leakage
4. Empty knowledge base and below-threshold similarity are handled as distinct cases with different user-facing messages
5. Below-threshold refusals include a contextual upload hint (finance sheet, loan policy, handbook, etc.) and — where the metric is known — an industry benchmark card
6. Retrieved chunks are formatted with filename and page labels and injected into the system prompt
7. GPT-4 streams the answer at `temperature=0` with strict grounding instructions
8. After streaming completes, a second LLM call scores faithfulness; a token-level grounding score is computed locally
9. Inline `p.X` citations in the answer are cross-validated against retrieved source metadata; mismatches are flagged in the UI and eval log
10. The `done` SSE event delivers sources (grouped by document), confidence, faithfulness, grounding score, and any unsupported claims

---

## Evaluation & Anti-Hallucination

The **Evaluation tab** shows a live per-query log with:

| Metric | Method |
|---|---|
| **Confidence** | Cosine similarity of the top retrieved chunk |
| **Faithfulness** | LLM verifies every factual claim against context (0–1) |
| **Grounding** | % of meaningful answer words found verbatim in retrieved chunks |
| **Citation mismatch** | Inline `p.X` citations cross-validated against source metadata |
| **Unsupported claims** | Individual claim strings flagged by the faithfulness scorer |

Thresholds used for colour coding: Confidence ≥ 65% green / ≥ 50% amber / < 50% red. Faithfulness ≥ 85% / ≥ 60%. Grounding ≥ 70% / ≥ 50%.

---

## Security & Tenant Isolation

- Every database query that touches `chunks`, `documents`, `conversations`, or `messages` includes a hard `WHERE tenant_id = $X` clause — there is no code path that can return another tenant's data
- The `X-Tenant-Id` header is validated on every request; requests with missing or short headers are rejected with HTTP 401
- PII (SSNs, emails, card numbers, long numeric IDs) is masked before any content is embedded or stored
- The LLM is instructed never to reveal tenant IDs, chunk IDs, similarity scores, or system internals, and to decline requests for credentials or sensitive personal data

---

## Known Limitations

- **Static benchmarks** — industry benchmark data is hardcoded for ~8 common metrics. Niche metrics won't trigger a benchmark card. A production version should pull from a curated dataset or secondary retrieval index.
- **Single-tenant per session** — `VITE_TENANT_ID` is set at build/deploy time. Multi-tenant auth (e.g. JWT-based tenant resolution) would be needed for a production SaaS deployment.
- **No re-ranking model** — retrieval uses cosine similarity only. A cross-encoder re-ranker (e.g. `ms-marco-MiniLM`) would improve precision on ambiguous queries.
- **Table extraction is layout-dependent** — pdfplumber's line-based table finder works well on ruled tables but may miss borderless or colour-background tables in complex PDFs.
- **Greeting detection is pattern-based** — the regex + set approach covers common cases but won't catch all conversational openers (e.g. "What's good?"). A lightweight intent classifier would be more robust at scale.