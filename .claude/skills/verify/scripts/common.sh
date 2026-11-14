# Sourced by doctor.sh and drive.sh. Resolves the repo, the binary, the checkpoint and the goldens.
# shellcheck shell=bash

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BIN="$REPO/.build/release/ming-tts"
EVIDENCE_ROOT="${MING_VERIFY_DIR:-$REPO/.audit/verify}"

# A checkpoint is complete when it holds these two files. campplus.onnx is needed only to clone.
checkpoint_complete() { [[ -f "$1/model.safetensors" && -f "$1/tokenizer.json" ]]; }

# Prints the checkpoint directory and how it was found, or returns 1.
# Order: MING_WEIGHTS, the README's download path, a sibling ming-ref checkout.
# When none match, ming-tts itself still searches the Hugging Face cache.
resolve_weights() {
    local candidate
    if [[ -n "${MING_WEIGHTS:-}" ]]; then
        echo "$MING_WEIGHTS"
        return 0
    fi
    for candidate in "$HOME/models/Ming-3.0-tts-0.5B" "$REPO/../ming-ref/weights"; do
        if checkpoint_complete "$candidate"; then
            (cd "$candidate" && pwd)
            return 0
        fi
    done
    return 1
}

# Prints the directory of golden parity cases, or returns 1. They are not published.
resolve_goldens() {
    local candidate
    for candidate in "${MING_GOLDENS:-}" "$REPO/../ming-ref/goldens"; do
        if [[ -n "$candidate" && -d "$candidate" ]]; then
            (cd "$candidate" && pwd)
            return 0
        fi
    done
    return 1
}
