"""
Database helpers.

Async functions (asyncpg) are used by the FastAPI routes.
Sync functions (psycopg2) are used by the Modal GPU worker.
Both sets import their driver lazily so this module is safe to import in
either the API container (no psycopg2) or the worker container (no asyncpg).
"""
from __future__ import annotations

import os
import uuid
from typing import Any


# ---------------------------------------------------------------------------
# Async — FastAPI / asyncpg
# ---------------------------------------------------------------------------

async def create_pool():
    import asyncpg
    dsn = os.environ["DATABASE_URL"]
    return await asyncpg.create_pool(dsn, ssl="require", min_size=1, max_size=5)


async def insert_job(pool, job_id: uuid.UUID, input_keys: list[str]) -> None:
    await pool.execute(
        "INSERT INTO jobs (id, input_keys) VALUES ($1, $2)",
        job_id,
        input_keys,
    )


async def get_job(pool, job_id: uuid.UUID) -> dict[str, Any] | None:
    row = await pool.fetchrow("SELECT * FROM jobs WHERE id = $1", job_id)
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Sync — Modal GPU worker / psycopg2
# ---------------------------------------------------------------------------

def _sync_conn():
    import psycopg2
    return psycopg2.connect(os.environ["DATABASE_URL"])


def sync_get_job(job_id: str) -> dict[str, Any]:
    import psycopg2.extras
    with _sync_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"Job {job_id} not found in DB")
    return dict(row)


def sync_update_job(job_id: str, **fields: Any) -> None:
    """Update any subset of job columns plus updated_at = now()."""
    if not fields:
        return
    set_clause = ", ".join(f"{k} = %s" for k in fields)
    values = list(fields.values()) + [job_id]
    with _sync_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE jobs SET {set_clause}, updated_at = now() WHERE id = %s",
                values,
            )
        conn.commit()
