#!/usr/bin/env bash
# Run the full test suite under ./tests and stream logs to the terminal.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== Running tests via unittest discover (tests/)..."
python3 -m unittest discover -s tests -p "test*.py" -t . "$@"
