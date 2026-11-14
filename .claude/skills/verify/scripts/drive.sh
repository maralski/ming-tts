#!/usr/bin/env bash
# Runs the real ming-tts binary once and keeps the evidence.
#
#   .claude/skills/verify/scripts/drive.sh <name> -- <ming-tts arguments>
#
#   drive.sh emma     -- --say "The garden stays open until sunset."
#   drive.sh sad      -- --say "I missed the last train." --emotion sad
#   drive.sh bad-flag -- --say hi --emotion bogus          # exit 2 is the evidence
#   drive.sh parity   -- --parity "$MING_GOLDENS/emma_short"
#   drive.sh bench    -- --bench --repeat 3
#
# It adds --weights (see common.sh for the search order) and, for --say and --parity,
# --out <evidence dir>/out.wav, unless you pass them yourself. A --bench run holds the
# GPU lock through scripts/gpulock.py. Paths you pass are resolved from your own
# working directory.
#
# Evidence goes to .audit/verify/<UTC time>-<name>/ (override the root with MING_VERIFY_DIR):
#   command.txt     the exact command line
#   stdout.txt      what the tool printed
#   stderr.txt      its error output
#   exit.txt        its exit code
#   out.wav         the audio, with out.fragN.wav when the text split into fragments
#   wavstat.json    format, length, peak, RMS and edge silence of each wav
#   transcript.txt  whisper's reading of out.wav (English --say runs, when whisper is installed)
#
# The script exits with ming-tts's exit code and prints the evidence directory last.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ $# -lt 3 || "$2" != "--" ]]; then
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 64
fi
name="$1"
shift 2
args=("$@")

[[ -x "$BIN" ]] || { echo "no binary at $BIN; run: swift build -c release" >&2; exit 69; }

has() { local flag; for flag in "${args[@]}"; do [[ "$flag" == "$1" ]] && return 0; done; return 1; }

dir="$EVIDENCE_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-$name"
mkdir -p "$dir"

# Added flags go first, so a trailing flag that lacks its value still fails as the user would see it.
if (has --say || has --parity) && ! has --out; then args=(--out "$dir/out.wav" "${args[@]}"); fi
if ! has --weights && weights="$(resolve_weights)"; then args=(--weights "$weights" "${args[@]}"); fi

cmd=("$BIN" "${args[@]}")
if has --bench; then cmd=("$REPO/scripts/gpulock.py" -- "${cmd[@]}"); fi

printf '%q ' "${cmd[@]}" > "$dir/command.txt"
echo >> "$dir/command.txt"

"${cmd[@]}" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
code=$?
echo "$code" > "$dir/exit.txt"

cat "$dir/stdout.txt"
cat "$dir/stderr.txt" >&2

shopt -s nullglob
wavs=("$dir"/*.wav)
if [[ ${#wavs[@]} -gt 0 ]]; then
    {
        echo "{"
        for i in "${!wavs[@]}"; do
            [[ $i -gt 0 ]] && echo ","
            printf '"%s": ' "$(basename "${wavs[$i]}")"
            "$(dirname "${BASH_SOURCE[0]}")/wavstat.py" "${wavs[$i]}"
        done
        echo "}"
    } > "$dir/wavstat.json"
    echo "--- wavstat"
    cat "$dir/wavstat.json"
fi

# Transcribe English speech so the proof covers the words, not only the waveform.
english=1
if has --dialect; then english=0; fi
for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "--language" ]]; then
        case "${args[$((i + 1))]:-}" in en|English|english) ;; *) english=0 ;; esac
    fi
done
if has --say && [[ $code -eq 0 && $english -eq 1 && -f "$dir/out.wav" && -z "${MING_VERIFY_NO_ASR:-}" ]] \
    && command -v whisper >/dev/null && [[ -f "$HOME/.cache/whisper/small.en.pt" ]]; then
    asr="$(mktemp -d)"
    if whisper "$dir/out.wav" --model small.en --language en --fp16 False \
        --output_format txt --output_dir "$asr" >/dev/null 2>&1; then
        cp "$asr/out.txt" "$dir/transcript.txt"
        echo "--- transcript"
        cat "$dir/transcript.txt"
    else
        echo "--- transcript skipped: whisper failed"
    fi
    rm -rf "$asr"
fi

echo "--- exit $code, evidence in $dir"
exit $code
