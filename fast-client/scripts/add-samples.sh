#!/usr/bin/env bash
# Insert demo samples via Supabase CLI.
# Requires: supabase CLI installed and local stack running.
set -euo pipefail

if ! command -v supabase >/dev/null 2>&1; then
  echo "Supabase CLI not found" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

supabase db query < "$PROJECT_ROOT/supabase/seed.sql"

echo "Inserted demo samples."
