---
name: verify
description: Drive the real ming-tts command-line tool (the Swift and Metal port of Ming-3.0-tts-0.5B) and capture proof that it speaks correctly. Use after any change to Sources/ to prove behavior on the real binary and real weights: speech from text, task flags (voice, style, emotion, dialect), voice cloning, golden parity, benchmarks and error messages. Also use when asked to verify, demo or reproduce something in MingTTS.
---

# Verify MingTTS

MingTTS has two surfaces. The primary one is the `ming-tts` command-line tool: it loads a 3 GB checkpoint, generates speech on the GPU and writes a WAV. The second is the `MingTTS` Swift library, which the tool wraps; drive it through the tool, and use `swift test` (see the end) for library-only API.

There is no server. Each run is one short process (about 1 s to load, then 5 times faster than real time), so "launch" means build once, then run the binary once per check.

All commands run from the repository root. `V` stands for `.claude/skills/verify/scripts`.

## Launch

1. Build the release binary. Debug builds are too slow to judge speed and fail timing tests.

   ```sh
   swift build -c release
   ```

   Ready when it prints `Build complete!` and `.build/release/ming-tts` exists. Rebuild after every source change; the doctor fails a stale binary.

2. Have a checkpoint. The helpers look, in order, at `MING_WEIGHTS`, `~/models/Ming-3.0-tts-0.5B`, and `../ming-ref/weights`. With none of those, `ming-tts` searches the Hugging Face cache. The weights are not in the repository; the README's "Get the weights" section has the download command. Do not download 3 GB without asking the user.

There is nothing to tear down: every run exits by itself.

## Doctor

```sh
.claude/skills/verify/scripts/doctor.sh
```

Read-only. It exits 0 when the machine is Apple silicon, the binary exists and is newer than every file in `Sources/` and `Package.swift`, and a complete checkpoint is found. It warns, without failing, when `campplus.onnx` is missing (cloning cannot run), when no goldens are found (parity cannot run), when `whisper` or its `small.en` model is missing (no transcript evidence), and when another `ming-tts` from this checkout is running (timings will be skewed). Run it first, and again whenever a result looks wrong.

## Drive

Every check goes through one helper, which runs the real binary and files the evidence:

```sh
.claude/skills/verify/scripts/drive.sh <name> -- <ming-tts arguments>
```

`<name>` labels the evidence directory. Everything after `--` is passed to `ming-tts` unchanged. The helper adds `--weights` and, for `--say` and `--parity`, `--out <evidence dir>/out.wav`, unless you pass them. It exits with the tool's exit code.

```sh
$V/drive.sh emma     -- --say "Meet at 3:30pm on Nov 6. It costs \$12.50."
$V/drive.sh captain  -- --say "Hoist the mainsail before the storm arrives." --style "a gravelly old sea captain"
$V/drive.sh conflict -- --say hi --plain --emotion sad        # expect exit 2
$V/drive.sh parity   -- --parity ../ming-ref/goldens/emma_short
$V/drive.sh bench    -- --bench --repeat 3
```

The stable handles are the flags and the lines the tool prints. The full flag list is the usage comment at the top of `Sources/ming-tts/main.swift`; `ming-tts` with no arguments prints a one-line usage, and there is no `--help`.

| Line on stdout | Meaning |
| --- | --- |
| `loaded <dir> in N s` | The checkpoint loaded. |
| `fragment: N steps, stopped true, prefill N ms` | One line per text fragment. `stopped true` means the model ended the utterance itself. `stopped false` means it ran to the step limit, which is a failure for normal text. |
| `N s of audio in N s wall (RTF N), first audio after N ms, peak N` | Totals. `peak 0.000` is silence. |
| `wrote <path>` | A WAV was written. Text that splits into several fragments also writes `out.fragN.wav`. |
| `PARITY PASS` or `PARITY FAIL` | Result of `--parity`, with one `ok` or `FAIL` line per tensor. |
| `ming-tts: bad input: ...` or `ming-tts: bad checkpoint: ...` on stderr, exit 2 | A rejected command line or a bad weights folder. |

Exit codes: 0 success, 1 parity failed, 2 error.

Two runs can go side by side: there are no ports and no shared files, and each evidence directory is named by time and `<name>`, so give concurrent runs different names. They share the GPU, so never read timings from overlapping runs. `drive.sh` runs `--bench` under `scripts/gpulock.py`, which serializes benchmarks across processes.

Pick the recipe for the feature you changed from [features/README.md](features/README.md). A proof that drives one entry point is incomplete when the feature file lists others your change touches.

## Evidence

Each run leaves a directory `.audit/verify/<UTC time>-<name>/` (the repository ignores `.audit/`; set `MING_VERIFY_DIR` to move the root):

| File | Content |
| --- | --- |
| `command.txt` | The exact command line. |
| `stdout.txt`, `stderr.txt`, `exit.txt` | What the tool printed and its exit code. |
| `out.wav`, `out.fragN.wav` | The audio: 32-bit float, mono, 44,100 Hz. |
| `wavstat.json` | Format, length, peak, RMS, non-finite sample count and silence at each end, per WAV. |
| `transcript.txt` | Whisper's reading of `out.wav`. Only for English `--say` runs, and only when `whisper` and its cached `small.en` model exist. Set `MING_VERIFY_NO_ASR=1` to skip it. |

Proof standards:

- Drive the built binary with real weights. A passing unit test is not proof that the tool speaks.
- A WAV on disk proves little. Speech is proven by all of: `stopped true` on every fragment, a `wavstat.json` with a plausible length (about 0.08 s per step), `peak` between roughly 0.1 and 1.0 and `nonfinite_samples` 0, and a `transcript.txt` that matches the input words. Whisper respells numbers and punctuation, so compare words, not characters.
- A transcript proves the words only. Claims about how it sounds (the voice, an emotion, a clone's likeness, a sound tag) need a listener: send `out.wav` to the user and say that it was not judged by ear.
- For a behavior change, capture a run from before and after the change, or a run of each branch of the behavior, not only the final state.
- For an error path, the evidence is `exit.txt` and `stderr.txt`, plus the absence of `out.wav`.
- Report the evidence directory path with every claim.

## Cleanup

- Processes: none outlive a run. If you interrupted one, kill the PID you started; never `pkill ming-tts`. `doctor.sh` lists any `ming-tts` still running from this checkout.
- Scratch: delete only files you made outside the evidence root, such as a trimmed reference clip in a temporary directory.
- Keep `.audit/verify/`. It is the proof. Leave `/tmp/ming-tts-gpu.lock` alone; other benchmark runs share it.

## Helpers

| Script | Use |
| --- | --- |
| `$V/doctor.sh` | The read-only readiness check above. |
| `$V/drive.sh <name> -- <args>` | Run `ming-tts` and file the evidence. Run it with no arguments for its usage text. |
| `$V/wavstat.py <file.wav>` | Print one WAV's facts as JSON. Use it on a reference clip before cloning it. |
| `$V/common.sh` | Sourced by the two shell helpers; holds the checkpoint and goldens search order. Not run directly. |

## The library surface

For API that the tool does not reach (`MingCheckpoint.missingFiles`, `MingVoice.catalog`, `MingReferenceVoice` coding, `trimSilence`), run the test suite the README documents:

```sh
MING_WEIGHTS=<checkpoint dir> MING_GOLDENS=<goldens dir> swift test -c release -Xswiftc -enable-testing
```

Tests that need `MING_WEIGHTS` or `MING_GOLDENS` are skipped when the variable is unset, so a green run without them proves less than it appears to. Check the skip count.
