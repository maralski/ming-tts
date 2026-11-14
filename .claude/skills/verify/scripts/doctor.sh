#!/usr/bin/env bash
# Read-only check: is this checkout worth driving? Exits 0 when every required line is ok.
#   .claude/skills/verify/scripts/doctor.sh
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

fail=0
ok()   { echo "ok    $*"; }
warn() { echo "warn  $*"; }
bad()  { echo "FAIL  $*"; fail=1; }

[[ "$(uname -m)" == "arm64" ]] && ok "Apple silicon" || bad "needs Apple silicon, this is $(uname -m)"

if [[ -x "$BIN" ]]; then
    stale="$(cd "$REPO" && find Sources Package.swift -type f -newer "$BIN" | head -5)"
    if [[ -z "$stale" ]]; then
        ok "binary $BIN is newer than every source file"
    else
        bad "binary is older than these sources; run: swift build -c release"
        echo "$stale" | sed 's/^/        /'
    fi
else
    bad "no binary at $BIN; run: swift build -c release"
fi

if weights="$(resolve_weights)"; then
    if checkpoint_complete "$weights"; then
        ok "checkpoint $weights"
        [[ -f "$weights/campplus.onnx" ]] && ok "campplus.onnx present (voice cloning can run)" \
            || warn "no campplus.onnx in the checkpoint; --clone will fail"
    else
        bad "checkpoint $weights lacks model.safetensors or tokenizer.json"
    fi
else
    warn "no checkpoint in MING_WEIGHTS, ~/models/Ming-3.0-tts-0.5B or ../ming-ref/weights; ming-tts will search the Hugging Face cache and fail if it is not there"
fi

if goldens="$(resolve_goldens)"; then
    ok "goldens $goldens ($(ls "$goldens" | grep -vc '\.'  ) cases)"
else
    warn "no goldens (MING_GOLDENS or ../ming-ref/goldens); --parity cannot run"
fi

if command -v whisper >/dev/null && [[ -f "$HOME/.cache/whisper/small.en.pt" ]]; then
    ok "whisper with small.en (drive.sh will transcribe English output)"
else
    warn "no whisper with a cached small.en model; drive.sh will skip the transcript, so judge speech by wavstat only"
fi

running="$(pgrep -fl "$BIN" || true)"
[[ -z "$running" ]] && ok "no other ming-tts from this checkout is running" \
    || warn "ming-tts already running (timings will be skewed): $running"

echo "evidence root: $EVIDENCE_ROOT"
exit $fail
