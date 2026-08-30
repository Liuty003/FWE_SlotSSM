#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$repo_root/scripts/run_openvla_oft_python.sh" \
  scripts/audit_relational_oft_handoffs.py "$@"
