"""
routers/routers.py

Consolidates:
  upload_router   — POST /upload
  files_router    — GET/DELETE /files
  prompts_router  — GET /prompts
  chat_router     — POST /chat
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re

import asyncpg
from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.config import get_settings
from core.database import get_pool
from services.services import (
    Chunk,
    RetrievedChunk,
    chunk_page,
    embed_texts,
    mask_pii,
    retrieve,
    stream_answer,
    get_chat_client,
    score_faithfulness,
    grounding_score,
)
from services.extractor import extract_document

log = logging.getLogger(__name__)


# ── Shared dependency ─────────────────────────────────────────────────────────

def get_tenant(x_tenant_id: str = Header(...)) -> str:
    if not x_tenant_id or len(x_tenant_id) < 2:
        raise HTTPException(401, "Missing or invalid X-Tenant-Id header")
    return x_tenant_id


async def db() -> asyncpg.Pool:
    return await get_pool()


# ══════════════════════════════════════════════════════════════════════════════
# upload_router  — POST /upload
# ══════════════════════════════════════════════════════════════════════════════
upload_router = APIRouter(prefix="/upload", tags=["upload"])

_TAG_RULES = [
    (["loan", "policy", "rule", "guideline", "credit"],                          "Policy"),
    (["handbook", "manual", "guide", "onboard", "sop"],                          "Operations"),
    (["finance", "budget", "revenue", "sheet", "q1","q2","q3","q4","cac","ltv"], "Finance"),
    (["pitch", "deck", "investor", "teaser"],                                    "Pitch"),
]


def _infer_tag(filename: str) -> str:
    fn = filename.lower()
    for keywords, tag in _TAG_RULES:
        if any(k in fn for k in keywords):
            return tag
    return "Document"


async def _generate_prompts(
    tenant_id: str, doc_id: int, tag: str, sample: str, pool: asyncpg.Pool
):
    cfg = get_settings()
    client = get_chat_client()
    try:
        resp = await client.chat.completions.create(
            model=cfg.azure_chat_deployment,
            max_tokens=300,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate exactly 3 short, practical questions a business founder "
                        "would ask about this document. "
                        "Return ONLY a valid JSON array of 3 strings. No markdown, no explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Document type: {tag}\n\nSample content:\n{sample[:2000]}",
                },
            ],
        )
        raw = re.sub(r"```json|```", "", resp.choices[0].message.content).strip()
        prompts: list[str] = json.loads(raw)
        async with pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO prompt_suggestions(tenant_id, doc_id, prompt, tag) VALUES($1,$2,$3,$4)",
                [(tenant_id, doc_id, p, tag) for p in prompts[:3]],
            )
    except Exception as e:
        log.warning("Prompt generation failed: %s", e)


@upload_router.post("")
async def upload_file(
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant),
    pool: asyncpg.Pool = Depends(db),
):
    cfg = get_settings()
    data = await file.read()
    file_hash = hashlib.sha256(data).hexdigest()

    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM documents WHERE tenant_id=$1 AND file_hash=$2",
            tenant_id, file_hash,
        )
        if existing:
            return {"status": "duplicate", "doc_id": existing["id"]}

        pages = extract_document(file.filename, data)
        tag = _infer_tag(file.filename)

        doc_id = await conn.fetchval(
            """INSERT INTO documents(tenant_id, filename, file_hash, tag, page_count)
               VALUES($1,$2,$3,$4,$5) RETURNING id""",
            tenant_id, file.filename, file_hash, tag, len(pages),
        )

        all_chunks: list[Chunk] = []
        for page in pages:
            all_chunks.extend(
                chunk_page(
                    page_num=page.page_num,
                    prose=page.text,
                    tables=page.tables,
                    chunk_size=cfg.chunk_size,
                    chunk_overlap=cfg.chunk_overlap,
                )
            )

        texts = [mask_pii(c.content) for c in all_chunks]
        embeddings = await embed_texts(texts)

        await conn.executemany(
            """INSERT INTO chunks(doc_id, tenant_id, page_num, chunk_index, chunk_type, content, embedding)
               VALUES($1,$2,$3,$4,$5,$6,$7::vector)""",
            [
                (doc_id, tenant_id, c.page_num, c.chunk_index,
                 c.chunk_type, mask_pii(c.content), str(emb))
                for c, emb in zip(all_chunks, embeddings)
            ],
        )
        await conn.execute(
            "UPDATE documents SET chunk_count=$1 WHERE id=$2",
            len(all_chunks), doc_id,
        )

    sample = pages[0].full_text if pages else ""
    # Fire-and-forget — don't block the upload response on LLM calls
    asyncio.create_task(_generate_prompts(tenant_id, doc_id, tag, sample, pool))

    return {
        "status": "ok",
        "doc_id": doc_id,
        "filename": file.filename,
        "tag": tag,
        "pages": len(pages),
        "chunks": len(all_chunks),
        "tables_found": sum(len(p.tables) for p in pages),
    }


# ══════════════════════════════════════════════════════════════════════════════
# files_router  — GET /files, DELETE /files/{doc_id}
# ══════════════════════════════════════════════════════════════════════════════
files_router = APIRouter(prefix="/files", tags=["files"])


@files_router.get("")
async def list_files(
    tenant_id: str = Depends(get_tenant),
    pool: asyncpg.Pool = Depends(db),
):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, filename, tag, page_count, chunk_count, created_at
               FROM documents WHERE tenant_id=$1 ORDER BY created_at DESC""",
            tenant_id,
        )
    return [dict(r) for r in rows]


@files_router.delete("/{doc_id}")
async def delete_file(
    doc_id: int,
    tenant_id: str = Depends(get_tenant),
    pool: asyncpg.Pool = Depends(db),
):
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM documents WHERE id=$1 AND tenant_id=$2",
            doc_id, tenant_id,
        )
    if result == "DELETE 0":
        raise HTTPException(404, "File not found or not yours")
    return {"status": "deleted", "doc_id": doc_id}


# ══════════════════════════════════════════════════════════════════════════════
# prompts_router  — GET /prompts
# ══════════════════════════════════════════════════════════════════════════════
prompts_router = APIRouter(prefix="/prompts", tags=["prompts"])


@prompts_router.get("")
async def get_prompts(
    tenant_id: str = Depends(get_tenant),
    pool: asyncpg.Pool = Depends(db),
):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT prompt, tag FROM prompt_suggestions
               WHERE tenant_id=$1 ORDER BY id DESC LIMIT 12""",
            tenant_id,
        )
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# chat_router  — POST /chat
# ══════════════════════════════════════════════════════════════════════════════
chat_router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


# ── Greeting detector ─────────────────────────────────────────────────────────
_GREETINGS = {
    "hi", "hey", "hello", "howdy", "hiya", "sup", "yo",
    "good morning", "good afternoon", "good evening", "good day",
    "how are you", "how's it going", "what's up", "whats up",
    "thanks", "thank you", "cheers", "bye", "goodbye", "see you",
    "ok", "okay", "cool", "got it", "sounds good", "perfect", "great",
}

_GREETING_RE = re.compile(
    r"^[\W]*(hi+|hey+|hello+|howdy|hiya|sup|yo|thanks?|thank you|"
    r"good\s+(morning|afternoon|evening|day)|how are you|"
    r"what'?s up|ok+a?y?|cool|got it|sounds good|perfect|great|"
    r"bye+|goodbye|see you|cheers)[\W]*$",
    re.IGNORECASE,
)


def _is_greeting(text: str) -> bool:
    s = text.strip().lower()
    return s in _GREETINGS or bool(_GREETING_RE.match(s))


# ── Follow-up detector ────────────────────────────────────────────────────────
# These phrases have no retrieval keywords — we re-use the prior question
# for retrieval so the same chunks are returned, but pass the actual
# follow-up text to the LLM so it answers in the right register.

_FOLLOWUP_RE = re.compile(
    r"^[\W]*(tell me more|more|elaborate|continue|go on|expand|explain more|"
    r"can you elaborate|what else|and\??|so\??|yes please|please continue|"
    r"keep going|what about that|say more|details please|more details|"
    r"give me more|can you expand|please elaborate|carry on)[\W]*$",
    re.IGNORECASE,
)


def _is_followup(text: str) -> bool:
    return bool(_FOLLOWUP_RE.match(text.strip()))


def _get_last_user_question(history: list[dict]) -> str | None:
    """Walk history backwards to find the last substantive user question."""
    for turn in reversed(history or []):
        if turn.get("role") == "user":
            q = turn.get("content", "").strip()
            if q and not _is_followup(q) and not _is_greeting(q):
                return q
    return None


# ── Contextual refusal hints ──────────────────────────────────────────────────
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


def _refusal_hint(question: str) -> str:
    q = question.lower()
    for keywords, hint in _REFUSAL_HINTS:
        if any(k in q for k in keywords):
            return hint
    return "Try uploading the relevant document — Horo searches only your own files."


def _no_docs_response(suggestion: str) -> dict:
    return {
        "answer":     "I don't have this in your uploaded documents.",
        "sources":    [],
        "confidence": 0.0,
        "missing":    True,
        "suggestion": suggestion,
    }


def _dedupe_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    seen: dict[tuple, float] = {}
    for c in chunks:
        key = (c.filename, c.page_num)
        if key not in seen or c.similarity > seen[key]:
            seen[key] = c.similarity
    return [
        {"doc": fn, "page": pg, "similarity": round(sim, 3)}
        for (fn, pg), sim in seen.items()
    ]


# ── Chat endpoint ─────────────────────────────────────────────────────────────
@chat_router.post("")
async def chat(
    req: ChatRequest,
    tenant_id: str = Depends(get_tenant),
    pool: asyncpg.Pool = Depends(db),
):
    cfg = get_settings()

    if not req.question.strip():
        raise HTTPException(400, "Empty question")

    # 1. Greeting short-circuit — no embed, no retrieval
    if _is_greeting(req.question):
        log.info("Greeting detected — skipping RAG pipeline")
        return {
            "answer": (
                "Hi! I'm Horo, your knowledge co-pilot. "
                "Ask me anything about your uploaded documents — "
                "policies, handbooks, finance sheets, or pitch decks — "
                "and I'll find the answer with exact source references."
            ),
            "sources":    [],
            "confidence": 1.0,
            "missing":    False,
            "greeting":   True,
        }

    # 2. Follow-up detection — retrieve on the prior substantive question
    #    so "tell me more" gets the same chunks as the original question.
    #    The actual follow-up text is still passed to the LLM.
    retrieval_query = req.question
    if _is_followup(req.question):
        prior = _get_last_user_question(req.history)
        if prior:
            log.info("Follow-up '%s' — retrieving on: %s", req.question, prior)
            retrieval_query = prior

    # 3. Embed retrieval query
    q_embs = await embed_texts([retrieval_query])
    q_emb  = q_embs[0]

    # 4. Retrieve (tenant-scoped, with drop-off filter)
    chunks = await retrieve(q_emb, tenant_id, pool, top_k=cfg.top_k)

    # 5. Empty knowledge base
    if not chunks:
        return _no_docs_response(
            "Your knowledge base is empty — upload a document to get started."
        )

    best = chunks[0].similarity

    # 6. Similarity gate — anti-hallucination hard stop
    if best < cfg.sim_threshold:
        return _no_docs_response(_refusal_hint(req.question))

    # 7. Stream answer + post-generation scoring
    async def event_stream():
        full_text = ""
        # Pass the ACTUAL question (not retrieval_query) so the LLM
        # answers "tell me more" in context, not re-answers the prior question
        async for token in stream_answer(req.question, chunks, req.history):
            full_text += token
            yield f"data: {token}\n\n"

        faithfulness, unsupported = await score_faithfulness(full_text, chunks)
        grounding = grounding_score(full_text, chunks)
        meta = {
            "sources":            _dedupe_sources(chunks),
            "confidence":         round(best, 3),
            "faithfulness":       faithfulness,
            "grounding_score":    grounding,
            "unsupported_claims": unsupported,
        }
        yield f"event: done\ndata: {json.dumps(meta)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")