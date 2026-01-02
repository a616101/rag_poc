#!/bin/bash
# PostgreSQL initialization script
# This script runs during the first startup of the PostgreSQL container
# and creates additional databases needed by the services.

set -e

echo "Creating additional databases..."

# Create litellm database for LiteLLM proxy
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create litellm database if not exists
    SELECT 'CREATE DATABASE litellm OWNER $POSTGRES_USER'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'litellm')\gexec

    -- Connect to litellm database and create extensions
    \c litellm
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";
EOSQL

echo "Additional databases created successfully!"
