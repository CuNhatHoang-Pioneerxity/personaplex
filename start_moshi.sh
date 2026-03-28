#!/bin/bash
# Start Moshi (Original PersonaPlex)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export HF_TOKEN=${HF_TOKEN:-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}
python3 -m moshi.server --host 0.0.0.0 --port 8998 --static ./client/dist
