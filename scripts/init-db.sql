-- GraphRAG PostgreSQL Extensions & Triggers
-- Table schemas are managed by SQLAlchemy (see models/sqlalchemy/)

-- ============================================================
-- Required Extensions
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- Trigger Function for updated_at
-- ============================================================
-- This trigger automatically updates the updated_at column
-- whenever a row is modified

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- ============================================================
-- Note: Table Creation
-- ============================================================
-- Tables are created by SQLAlchemy's Base.metadata.create_all()
-- during application startup (see db/connection.py).
--
-- Schema migrations (adding missing columns, renaming columns,
-- fixing ENUM types) are handled by the _migrate_existing_tables()
-- function in db/connection.py.
--
-- Do NOT define tables here to avoid schema conflicts.
-- ============================================================
