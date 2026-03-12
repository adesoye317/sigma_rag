"""
Patches and improvements over the reviewed codebase.
Each section is labelled with the issue it fixes.
"""

# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 — conversations.py: register the delete route + move /eval above /{id}
# ══════════════════════════════════════════════════════════════════════════════
# FastAPI matches routes top-to-bottom. A parameterised segment like /{conv_id}
# will swallow the literal string "eval" and try int("eval") → HTTP 422.
# Rule: always register specific literal paths BEFORE parameterised ones.

from __future__ import annotations
import asyncio
import json
import logging
import re
from typing import AsyncIterator

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.database import get_pool
from routers.routers import get_tenant

log = logging.getLogger(__name__)
conversations_router = APIRouter(prefix="/conversations", tags=["conversations"])


async def db() -> asyncpg.Pool:
    return await get_pool()


class ConversationCreate(BaseModel):
    title: str = "New Chat"


class MessageSave(BaseModel):
    role: str
    content: str
    sources: list[dict] = []
    confidence: float | None = None
    faithfulness: float | None = None
    grounding_score: float | None = None
    unsupported_claims: list[str] = []


@conversations_router.get("")
async def list_conversations(
    tenant_id: str = Depends(get_tenant),
    pool: asyncpg.Pool = Depends(db),
):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.tenant_id = $1
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT 50
            """,
            tenant_id,
        )
    return [dict(r) for r in rows]


@conversations_router.post("")
async def create_conversation(
    body: ConversationCreate,
    tenant_id: str = Depends(get_tenant),
    pool: asyncpg.Pool = Depends(db),
):
    async with pool.acquire() as conn:
        conv_id = await conn.fetchval(
            "INSERT INTO conversations(tenant_id, title) VALUES($1,$2) RETURNING id",
            tenant_id, body.title,
        )
    return {"id": conv_id, "title": body.title}


# ── FIX 1: /eval MUST come before /{conv_id} ─────────────────────────────────
@conversations_router.get("/eval")
async def get_eval_log(
    tenant_id: str = Depends(get_tenant),
    pool: asyncpg.Pool = Depends(db),
):
    """Return paired question/answer rows with metrics for the eval tab."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH ordered AS (
                SELECT
                    m.id, m.conversation_id, m.role, m.content,
                    m.confidence, m.faithfulness, m.grounding_score,
                    m.unsupported_claims, m.sources,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.conversation_id ORDER BY m.created_at
                    ) AS rn
                FROM messages m
                WHERE m.tenant_id = $1
            ),
            pairs AS (
                SELECT
                    a.content AS answer, u.content AS question,
                    a.confidence, a.faithfulness, a.grounding_score,
                    a.unsupported_claims, a.sources, a.id AS msg_id
                FROM ordered a
                JOIN ordered u
                  ON u.conversation_id = a.conversation_id
                 AND u.rn = a.rn - 1
                 AND u.role = 'user'
                WHERE a.role = 'assistant'
                  AND a.confidence IS NOT NULL
            )
            SELECT * FROM pairs ORDER BY msg_id DESC LIMIT 200
            """,
            tenant_id,
        )
    return [
        {
            "question":           r["question"],
            "confidence":         r["confidence"] or 0,
            "faithfulness":       r["faithfulness"],
            "grounding_score":    r["grounding_score"],
            "unsupported_claims": json.loads(r["unsupported_claims"]) if r["unsupported_claims"] else [],
            "source_count":       len(json.loads(r["sources"])) if r["sources"] else 0,
            "missing":            (r["confidence"] or 0) < 0.45,
        }
        for r in rows
    ]


@conversations_router.get("/{conv_id}/messages")
async def get_messages(
    conv_id: int,
    tenant_id: str = Depends(get_tenant),
    pool: asyncpg.Pool = Depends(db),
):
    async with pool.acquire() as conn:
        owner = await conn.fetchval(
            "SELECT id FROM conversations WHERE id=$1 AND tenant_id=$2",
            conv_id, tenant_id,
        )
        if not owner:
            raise HTTPException(404, "Conversation not found")
        rows = await conn.fetch(
            """
            SELECT id, role, content, sources, confidence,
                   faithfulness, grounding_score, unsupported_claims, created_at
            FROM messages WHERE conversation_id = $1 ORDER BY created_at ASC
            """,
            conv_id,
        )
    return [
        {
            **dict(r),
            "sources":            json.loads(r["sources"])            if r["sources"]            else [],
            "unsupported_claims": json.loads(r["unsupported_claims"]) if r["unsupported_claims"] else [],
        }
        for r in rows
    ]


@conversations_router.post("/{conv_id}/messages")
async def save_message(
    conv_id: int,
    body: MessageSave,
    tenant_id: str = Depends(get_tenant),
    pool: asyncpg.Pool = Depends(db),
):
    async with pool.acquire() as conn:
        owner = await conn.fetchval(
            "SELECT id FROM conversations WHERE id=$1 AND tenant_id=$2",
            conv_id, tenant_id,
        )
        if not owner:
            raise HTTPException(404, "Conversation not found")

        await conn.execute(
            """
            INSERT INTO messages(
                conversation_id, tenant_id, role, content,
                sources, confidence, faithfulness, grounding_score, unsupported_claims
            ) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9)
            """,
            conv_id, tenant_id, body.role, body.content,
            json.dumps(body.sources), body.confidence,
            body.faithfulness, body.grounding_score,
            json.dumps(body.unsupported_claims),
        )
        if body.role == "user":
            await conn.execute(
                """
                UPDATE conversations
                SET title = CASE WHEN title = 'New Chat' THEN LEFT($1, 60) ELSE title END,
                    updated_at = NOW()
                WHERE id = $2
                """,
                body.content, conv_id,
            )
    return {"status": "saved"}


# ── FIX 1b: decorator was missing entirely ───────────────────────────────────
@conversations_router.delete("/{conv_id}")
async def delete_conversation(
    conv_id: int,
    tenant_id: str = Depends(get_tenant),
    pool: asyncpg.Pool = Depends(db),
):
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM conversations WHERE id=$1 AND tenant_id=$2",
            conv_id, tenant_id,
        )
    if result == "DELETE 0":
        raise HTTPException(404, "Conversation not found")
    return {"status": "deleted"}


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2 — upload.py: make _generate_prompts non-blocking
# ══════════════════════════════════════════════════════════════════════════════
# Original code awaited _generate_prompts directly, stalling the HTTP response
# for 3 sequential LLM calls (~3–6 s). Fire it as a background task instead.

async def _generate_prompts_safe(tenant_id, doc_id, tag, sample, pool):
    """Wraps _generate_prompts with top-level error swallowing for fire-and-forget."""
    try:
        await _generate_prompts(tenant_id, doc_id, tag, sample, pool)
    except Exception as e:
        log.warning("Background prompt generation failed: %s", e)


# In upload_file, replace:
#   await _generate_prompts(tenant_id, doc_id, tag, sample, pool)
# with:
#   asyncio.create_task(_generate_prompts_safe(tenant_id, doc_id, tag, sample, pool))


# ══════════════════════════════════════════════════════════════════════════════
# FIX 3 — services.py: token-budgeted history + smarter refusal messages
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. History budget ────────────────────────────────────────────────────────
_MAX_HISTORY_TOKENS = 600   # ~450 words of prior conversation
_APPROX_CHARS_PER_TOKEN = 4


def _trim_history(history: list[dict], budget_tokens: int = _MAX_HISTORY_TOKENS) -> list[dict]:
    """
    Keep the most recent turns that fit within the token budget.
    Works backwards so we always preserve the most recent context.
    """
    budget = budget_tokens * _APPROX_CHARS_PER_TOKEN
    kept: list[dict] = []
    for turn in reversed(history or []):
        if turn.get("role") not in ("user", "assistant"):
            continue
        budget -= len(turn.get("content", ""))
        if budget < 0:
            break
        kept.append(turn)
    return list(reversed(kept))


# ── 3b. Contextual refusal message ───────────────────────────────────────────
_REFUSAL_HINTS: list[tuple[list[str], str]] = [
    (
        ["cac", "ltv", "revenue", "budget", "mrr", "arr", "burn", "runway",
         "finance", "p&l", "profit", "loss", "sheet"],
        "Try uploading your latest **finance sheet** or **P&L** (Excel or PDF).",
    ),
    (
        ["loan", "policy", "rate", "limit", "eligibility", "terms", "condition",
         "repayment", "collateral", "borrow"],
        "Try uploading your **loan policy** or **credit guidelines** document.",
    ),
    (
        ["onboard", "step", "process", "procedure", "handbook", "sop", "manual",
         "guide", "workflow"],
        "Try uploading your **operations handbook** or **SOP** document.",
    ),
    (
        ["pitch", "deck", "investor", "valuation", "raise", "round"],
        "Try uploading your **pitch deck** or **investor materials**.",
    ),
    (
        ["employee", "hr", "leave", "benefit", "salary", "payroll", "contract"],
        "Try uploading your **HR handbook** or **employment contracts**.",
    ),
]

_REFUSAL_DEFAULT = (
    "Try uploading the relevant document — "
    "Horo searches only your own files."
)


def _refusal_hint(question: str) -> str:
    q = question.lower()
    for keywords, hint in _REFUSAL_HINTS:
        if any(k in q for k in keywords):
            return hint
    return _REFUSAL_DEFAULT


# ── 3c. Updated stream_answer with trimmed history ───────────────────────────
async def stream_answer_v2(
    question: str,
    chunks,                     # list[RetrievedChunk]
    history: list[dict],
) -> AsyncIterator[str]:
    from core.config import get_settings
    cfg = get_settings()
    client = get_chat_client()
    context = _build_context(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in _trim_history(history):
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {question}",
    })

    stream = await client.chat.completions.create(
        model=cfg.azure_chat_deployment,
        messages=messages,
        max_tokens=700,
        temperature=0,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


# ── 3d. Updated chat route using the fixes above ─────────────────────────────
from fastapi.responses import StreamingResponse
from pydantic import BaseModel as _BaseModel

chat_router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(_BaseModel):
    question: str
    history: list[dict] = []


@chat_router.post("")
async def chat(
    req: ChatRequest,
    tenant_id: str = Depends(get_tenant),
    pool: asyncpg.Pool = Depends(db),
):
    from core.config import get_settings
    from services.services import (
        embed_texts, retrieve, score_faithfulness,
        grounding_score as grounding_score_fn,
    )
    cfg = get_settings()
    if not req.question.strip():
        raise HTTPException(400, "Empty question")

    q_embs = await embed_texts([req.question])
    chunks = await retrieve(q_embs[0], tenant_id, pool, top_k=cfg.top_k)

    if not chunks:
        return {
            "answer": "I don't have this in your uploaded documents.",
            "sources": [], "confidence": 0.0, "missing": True,
            # FIX 3b: contextual hint instead of one-size-fits-all message
            "suggestion": _refusal_hint(req.question),
        }

    best = chunks[0].similarity
    if best < cfg.sim_threshold:
        return {
            "answer": "I don't have this in your uploaded documents.",
            "sources": [], "confidence": 0.0, "missing": True,
            "suggestion": _refusal_hint(req.question),
        }

    async def event_stream():
        full_text = ""
        # FIX 3c: use budgeted-history version
        async for token in stream_answer_v2(req.question, chunks, req.history):
            full_text += token
            yield f"data: {token}\n\n"

        faithfulness, unsupported = await score_faithfulness(full_text, chunks)
        grounding = grounding_score_fn(full_text, chunks)
        meta = {
            "sources":            _dedupe_sources(chunks),
            "confidence":         round(best, 3),
            "faithfulness":       faithfulness,
            "grounding_score":    grounding,
            "unsupported_claims": unsupported,
        }
        yield f"event: done\ndata: {json.dumps(meta)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _dedupe_sources(chunks) -> list[dict]:
    seen: dict[tuple, float] = {}
    for c in chunks:
        key = (c.filename, c.page_num)
        if key not in seen or c.similarity > seen[key]:
            seen[key] = c.similarity
    return [
        {"doc": fn, "page": pg, "similarity": round(sim, 3)}
        for (fn, pg), sim in seen.items()
    ]


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT — re-rank chunks before feeding to LLM
# ══════════════════════════════════════════════════════════════════════════════
# Pure vector similarity sometimes surfaces off-topic chunks when the query is
# short or ambiguous.  A lightweight BM25-style keyword boost applied on top of
# the cosine scores improves precision without an extra API call.

def _keyword_boost(
    question: str,
    chunks,                   # list[RetrievedChunk]
    weight: float = 0.25,
) -> list:                    # list[RetrievedChunk], re-ordered
    """
    Boost cosine similarity by the fraction of question keywords that appear
    in each chunk, then re-sort descending.  No external libraries needed.
    """
    stop = {"what", "is", "the", "are", "a", "an", "of", "in", "for", "my", "our"}
    q_words = {w for w in re.findall(r"\b\w{3,}\b", question.lower()) if w not in stop}
    if not q_words:
        return chunks

    boosted = []
    for c in chunks:
        chunk_text = c.content.lower()
        hit_rate = sum(1 for w in q_words if w in chunk_text) / len(q_words)
        boosted.append((c, c.similarity + weight * hit_rate))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in boosted]


# Usage: insert before the similarity gate in the chat handler
#   chunks = _keyword_boost(req.question, chunks)