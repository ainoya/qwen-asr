#!/usr/bin/env bash
# tools/setup-benchmark-hook.sh - Configure git to use repository hooks in .githooks
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR"
git config core.hooksPath .githooks
chmod +x .githooks/*
echo "Successfully configured git core.hooksPath to .githooks"
