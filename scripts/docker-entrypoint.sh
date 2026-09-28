#!/bin/sh
set -eu

attempt=1
max_attempts="${MIGRATION_MAX_ATTEMPTS:-10}"
retry_seconds="${MIGRATION_RETRY_SECONDS:-2}"

until alembic upgrade head; do
    if [ "$attempt" -ge "$max_attempts" ]; then
        echo "database migration failed after ${attempt} attempts" >&2
        exit 1
    fi
    echo "database migration attempt ${attempt} failed; retrying in ${retry_seconds}s" >&2
    attempt=$((attempt + 1))
    sleep "$retry_seconds"
done

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
