#!/bin/bash
# Start Moshi (Original PersonaPlex)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure NVM is in PATH for node/npm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

cd "$SCRIPT_DIR"
export HF_TOKEN=${HF_TOKEN:-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}
python3 -m moshi.server --host 0.0.0.0 --port 8998 --static ./client/dist
