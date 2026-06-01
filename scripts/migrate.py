#!/usr/bin/env python3
"""
Apply SQL migrations to the configured DATABASE_URL.

Usage:
    export DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
    python scripts/migrate.py

Migrations in migrations/ are applied in filename order. Each is idempotent
(uses IF NOT EXISTS) so re-running is safe.
"""
import os
import sys
from pathlib import Path

try:
    import psycopg2
except ImportError:
    sys.exit("psycopg2 not found — run: pip install psycopg2-binary")

dsn = os.environ.get("DATABASE_URL")
if not dsn:
    sys.exit("DATABASE_URL is not set")

migrations_dir = Path(__file__).parent.parent / "migrations"
migration_files = sorted(migrations_dir.glob("*.sql"))

if not migration_files:
    sys.exit(f"No .sql files found in {migrations_dir}")

conn = psycopg2.connect(dsn)
conn.autocommit = True
cur = conn.cursor()

for path in migration_files:
    sql = path.read_text()
    print(f"Applying {path.name} ...", end=" ", flush=True)
    cur.execute(sql)
    print("done")

cur.close()
conn.close()
print("Migrations complete.")
