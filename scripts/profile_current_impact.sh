#!/bin/bash
# Profile slopometry current-impact command using py-spy
#
# Usage:
#   ./scripts/profile_current_impact.sh /path/to/large/repo
#   ./scripts/profile_current_impact.sh /path/to/large/repo profile.json
#
# Requirements:
#   pip install py-spy  (or: uv tool install py-spy)
#
# Output formats:
#   - speedscope (.json) - open with https://speedscope.app
#   - flamegraph (.svg)  - open in browser

set -e

REPO_PATH="${1:-.}"
OUTPUT="${2:-profile.json}"

case "$OUTPUT" in
    *.svg)
        FORMAT="flamegraph"
        ;;
    *)
        FORMAT="speedscope"
        ;;
esac

echo "Profiling current-impact on: $REPO_PATH"
echo "Output format: $FORMAT"
echo "Output file: $OUTPUT"
echo ""

if ! command -v py-spy &> /dev/null; then
    echo "Error: py-spy not found. Install with:"
    echo "  pip install py-spy"
    echo "  # or"
    echo "  uv tool install py-spy"
    exit 1
fi

# Run profiling with py-spy
# --subprocesses: Also profile child processes (ProcessPoolExecutor workers)
# --native: Include native (C) frames for full picture
py-spy record \
    --output "$OUTPUT" \
    --format "$FORMAT" \
    --subprocesses \
    --native \
    -- uv run slopometry summoner current-impact \
        --repo-path "$REPO_PATH" \
        --recompute-baseline \
        --no-pager

echo ""
echo "Profile saved to: $OUTPUT"

if [ "$FORMAT" = "speedscope" ]; then
    echo "Open with: https://speedscope.app (drag and drop the file)"
    echo "Or install speedscope: npm install -g speedscope && speedscope $OUTPUT"
else
    echo "Open in browser: $OUTPUT"
fi
