import ssl
import asyncpg
from core.config import get_settings

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    return _pool


async def init_pool() -> asyncpg.Pool:
    global _pool
    cfg = get_settings()

    ssl_ctx: bool | ssl.SSLContext = False
    if cfg.db_ssl == "require":
        ssl_ctx = True
    elif cfg.db_ssl == "verify-full":
        import ssl as _ssl
        ssl_ctx = _ssl.create_default_context()

    _pool = await asyncpg.create_pool(
        host=cfg.db_host,
        port=cfg.db_port,
        database=cfg.db_name,
        user=cfg.db_user,
        password=cfg.db_password,
        min_size=cfg.db_pool_min,
        max_size=cfg.db_pool_max,
        ssl=ssl_ctx,
    )
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()


async def init_schema(pool: asyncpg.Pool):
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id          SERIAL PRIMARY KEY,
                tenant_id   TEXT NOT NULL,
                filename    TEXT NOT NULL,
                file_hash   TEXT NOT NULL,
                tag         TEXT,
                page_count  INT DEFAULT 0,
                chunk_count INT DEFAULT 0,
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(tenant_id, file_hash)
            )
        """)

        dim = get_settings().azure_embed_dimensions
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          SERIAL PRIMARY KEY,
                doc_id      INT REFERENCES documents(id) ON DELETE CASCADE,
                tenant_id   TEXT NOT NULL,
                page_num    INT,
                chunk_index INT,
                chunk_type  TEXT DEFAULT 'text',
                content     TEXT NOT NULL,
                embedding   vector({dim})
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_suggestions (
                id        SERIAL PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                doc_id    INT REFERENCES documents(id) ON DELETE CASCADE,
                prompt    TEXT NOT NULL,
                tag       TEXT
            )
        """)

        # ── Conversation history ──────────────────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         SERIAL PRIMARY KEY,
                tenant_id  TEXT NOT NULL,
                title      TEXT NOT NULL DEFAULT 'New Chat',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              SERIAL PRIMARY KEY,
                conversation_id INT REFERENCES conversations(id) ON DELETE CASCADE,
                tenant_id       TEXT NOT NULL,
                role            TEXT NOT NULL,   -- 'user' | 'assistant'
                content         TEXT NOT NULL,
                sources         JSONB DEFAULT '[]',
                confidence      FLOAT,
                faithfulness    FLOAT,
                grounding_score FLOAT,
                unsupported_claims JSONB DEFAULT '[]',
                created_at      TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON chunks USING hnsw (embedding vector_cosine_ops)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON chunks (tenant_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_tenant
            ON conversations (tenant_id, updated_at DESC)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages (conversation_id, created_at ASC)
        """)