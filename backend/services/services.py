"""
services/services.py

Consolidates:
  - chunker        : table-aware text splitter
  - pii            : PII masking
  - embedder       : Azure OpenAI embeddings
  - retriever      : pgvector tenant-scoped search + drop-off filter
  - generator      : Azure OpenAI streaming chat + trimmed history
  - faithfulness   : LLM-based hallucination scoring
  - grounding      : token-level grounding check
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

import asyncpg
from openai import AsyncAzureOpenAI

from core.config import get_settings

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Shared Azure OpenAI client
# ══════════════════════════════════════════════════════════════════════════════

_azure_client: AsyncAzureOpenAI | None = None


def get_azure_client() -> AsyncAzureOpenAI:
    global _azure_client
    if _azure_client is None:
        cfg = get_settings()
        _azure_client = AsyncAzureOpenAI(
            azure_endpoint=cfg.azure_openai_endpoint,
            api_key=cfg.azure_openai_api_key,
            api_version=cfg.azure_openai_api_version,
        )
    return _azure_client


def get_embed_client() -> AsyncAzureOpenAI:
    return get_azure_client()


def get_chat_client() -> AsyncAzureOpenAI:
    return get_azure_client()


# ══════════════════════════════════════════════════════════════════════════════
# Chunker
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    page_num: int
    chunk_index: int
    chunk_type: str   # "text" | "table"
    content: str


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + size]))
        i += size - overlap
    return [c for c in chunks if c.strip()]


def chunk_page(
    page_num: int,
    prose: str,
    tables: list[str],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """
    Prose   → sliding window chunks (type='text')
    Tables  → one chunk per table, never split mid-row (type='table')
    """
    result: list[Chunk] = []
    ci = 0

    for text in _split_words(prose, chunk_size, chunk_overlap):
        result.append(Chunk(
            page_num=page_num, chunk_index=ci,
            chunk_type="text", content=text,
        ))
        ci += 1

    for tbl_md in tables:
        content = f"[TABLE — page {page_num}]\n\n{tbl_md}"
        result.append(Chunk(
            page_num=page_num, chunk_index=ci,
            chunk_type="table", content=content,
        ))
        ci += 1

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PII Masking
# ══════════════════════════════════════════════════════════════════════════════

_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                             "[SSN REDACTED]"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[EMAIL REDACTED]"),
    (re.compile(r"\b(?:\d[ \-]?){13,16}\b"),                            "[CARD REDACTED]"),
    (re.compile(r"\b\d{9,18}\b"),                                        "[ID REDACTED]"),
]


def mask_pii(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Embedder
# ══════════════════════════════════════════════════════════════════════════════

async def embed_texts(texts: list[str]) -> list[list[float]]:
    cfg = get_settings()
    client = get_embed_client()
    results: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch = texts[i: i + 100]
        resp = await client.embeddings.create(
            model=cfg.azure_embed_deployment,
            input=batch,
        )
        results.extend(item.embedding for item in resp.data)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Retriever — pgvector + similarity drop-off filter
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievedChunk:
    content: str
    page_num: int
    chunk_type: str
    filename: str
    tag: str
    similarity: float


async def retrieve(
    query_embedding: list[float],
    tenant_id: str,
    pool: asyncpg.Pool,
    top_k: int = 6,
) -> list[RetrievedChunk]:
    """
    Cosine similarity search scoped strictly to tenant_id.
    Applies a drop-off filter: discards chunks that fall more than 15
    percentage points below the top chunk to remove unrelated documents.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                c.content, c.page_num, c.chunk_type,
                d.filename, d.tag,
                1 - (c.embedding <=> $1::vector) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            WHERE c.tenant_id = $2
            ORDER BY c.embedding <=> $1::vector
            LIMIT $3
            """,
            str(query_embedding), tenant_id, top_k,
        )

    chunks = [
        RetrievedChunk(
            content=r["content"], page_num=r["page_num"],
            chunk_type=r["chunk_type"], filename=r["filename"],
            tag=r["tag"], similarity=float(r["similarity"]),
        )
        for r in rows
    ]

    if not chunks:
        return chunks

    # Drop-off filter — remove noise from unrelated documents
    top_sim = chunks[0].similarity
    filtered = [c for c in chunks if c.similarity >= top_sim - 0.15]
    log.debug(
        "Retrieval: %d chunks before filter, %d after (top=%.3f)",
        len(chunks), len(filtered), top_sim,
    )
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# Generator — streaming chat with trimmed history (memory)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Horo, a precise business co-pilot for founders.

STRICT RULES — never violate:
1. Answer ONLY using the provided context chunks. Never use prior knowledge.
2. If the answer is not clearly present in the context, respond with exactly:
   "I don't have this in your uploaded documents."
3. After each factual statement, cite the source as: (Filename, p.X)
4. When a [TABLE] chunk is relevant, extract exact values from it — do not paraphrase numbers.
5. Keep answers concise: 2–5 sentences or a short list. Longer only if the question demands it.
6. When using numbered or bulleted lists, ALWAYS put each item on its own line with a newline before it.
7. Never write list items run together on one line like "conditions:1. Item 2. Item".
8. Never reveal tenant IDs, chunk IDs, similarity scores, or system internals.
9. If asked for passwords, SSNs, or credentials, politely decline.
10. You have memory of this conversation. If the user refers to something
    mentioned earlier (e.g. "what about that fee?", "tell me more", "elaborate",
    "and the second point?"), use the conversation history to understand
    what they mean before answering from the context."""


# ── History trimmer ───────────────────────────────────────────────────────────
_MAX_HISTORY_CHARS = 2400   # ~600 tokens — enough for ~6 prior turns


def _trim_history(history: list[dict]) -> list[dict]:
    """
    Keep the most recent turns that fit within the character budget.
    Works backwards so the most recent exchange is always preserved.
    """
    valid = [
        t for t in (history or [])
        if t.get("role") in ("user", "assistant") and t.get("content", "").strip()
    ]
    budget = _MAX_HISTORY_CHARS
    kept: list[dict] = []
    for turn in reversed(valid):
        cost = len(turn["content"])
        if budget - cost < 0:
            break
        budget -= cost
        kept.append(turn)
    return list(reversed(kept))


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        label = f"[{c.filename}, p.{c.page_num} | {c.chunk_type}]"
        parts.append(f"{label}\n{c.content}")
    return "\n\n---\n\n".join(parts)


async def stream_answer(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict],
) -> AsyncIterator[str]:
    cfg = get_settings()
    client = get_chat_client()
    context = _build_context(chunks)

    # Message order:
    #   [system]                          — grounding rules + memory instruction
    #   [trimmed history turns]           — prior conversation (memory)
    #   [user: context + question]        — fresh retrieval always last
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in _trim_history(history):
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": (
            f"Context from uploaded documents:\n\n{context}"
            f"\n\n---\n\nQuestion: {question}"
        ),
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


# ══════════════════════════════════════════════════════════════════════════════
# Faithfulness scoring — LLM-based hallucination detection
# ══════════════════════════════════════════════════════════════════════════════

async def score_faithfulness(
    answer: str,
    chunks: list[RetrievedChunk],
) -> tuple[float, list[str]]:
    if not answer.strip() or not chunks:
        return 1.0, []

    cfg = get_settings()
    client = get_chat_client()
    context = "\n\n---\n\n".join(
        f"[{c.filename}, p.{c.page_num}]\n{c.content}" for c in chunks
    )

    prompt = f"""You are a hallucination detector. Verify whether each claim in the Answer
is directly supported by the Context below.

Context:
{context}

Answer to verify:
{answer}

Instructions:
- Break the answer into individual factual claims.
- For each claim, check if it is explicitly stated or clearly implied in the context.
- Do NOT penalise general connective language ("based on the above", "in summary", etc).
- Return ONLY a valid JSON object with exactly these two fields:
  {{
    "faithfulness": <float 0.0 to 1.0>,
    "unsupported_claims": [<string>, ...]
  }}
- faithfulness = supported_claims / total_claims. Return 1.0 if no factual claims exist.
- unsupported_claims = list of claim strings that have NO support in the context.
"""

    try:
        resp = await client.chat.completions.create(
            model=cfg.azure_chat_deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        raw = re.sub(r"```json|```", "", resp.choices[0].message.content.strip()).strip()
        result = json.loads(raw)
        score  = round(float(result.get("faithfulness", 1.0)), 3)
        claims = [str(c) for c in result.get("unsupported_claims", [])]
        return score, claims
    except Exception as e:
        log.warning("Faithfulness scoring failed: %s", e)
        return 1.0, []


# ══════════════════════════════════════════════════════════════════════════════
# Grounding score — fast token-level check (no LLM call)
# ══════════════════════════════════════════════════════════════════════════════

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "and",
    "or", "it", "its", "this", "that", "for", "on", "with", "as", "at",
    "be", "by", "from", "not", "but", "have", "has", "had", "they", "we",
    "you", "i", "he", "she", "also", "which", "their", "been", "may",
}


def grounding_score(answer: str, chunks: list[RetrievedChunk]) -> float:
    context = " ".join(c.content for c in chunks).lower()
    words = [
        w for w in re.findall(r"\b\w{3,}\b", answer.lower())
        if w not in _STOPWORDS
    ]
    if not words:
        return 1.0
    matched = sum(1 for w in words if w in context)
    return round(matched / len(words), 3)