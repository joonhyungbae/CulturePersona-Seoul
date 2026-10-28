#!/usr/bin/env bash
# Local test server (the deploy is GitHub Pages + Apps Script). Storage is ../outputs/anno/ (the main repository, gitignore).
#   ./run_local.sh           127.0.0.1:5180
#   ANNO_STORE=/tmp/x ./run_local.sh   another store (for tests)
set -euo pipefail
cd "$(dirname "$0")"
[ -d node_modules ] || npm install
echo "Local annotation tool: http://127.0.0.1:5180/  (codes in ../outputs/anno/raters.json)"
exec npx vite
