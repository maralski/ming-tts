# Golden parity

`ming-tts --parity <case dir>` replays one case captured from the Python reference and compares the port's prompt, hidden states, latents and waveform with the captured tensors. It is the check that the port still computes what the reference model computes.

## Sub-features

- `parity-builtin` passes the built-in voice cases `emma_short`, `emma_long` and `george_short`.
- `parity-clone` passes the clone cases `clone_george` and `clone_lingguang`.
- `parity-exit` exits 0 on `PARITY PASS` and 1 on `PARITY FAIL`.
- `parity-wav` writes the end-to-end waveform to `--out`.

## How to get to it (user POV)

- `ming-tts --parity <case dir>`, with `--weights dir` or `MING_WEIGHTS`.
- Add `--out file.wav` to choose where the end-to-end audio goes; the default is `out.wav`.

## Driving it with drive.sh

Preconditions:

- `$V/doctor.sh` exits 0 and prints an `ok    goldens <dir>` line. The goldens are not published. With no goldens, report this feature as not run.
- `G` is the goldens directory from that line.

- **One case.** Run `$V/drive.sh parity-emma-short -- --parity "$G/emma_short"`. Exit 0. Stdout has `ok   prompt ids`, then one `ok` line per tensor (`inputs_embeds`, `llm_layer0_prefill`, `llm_layer23_prefill`, `llm_hidden`, `stop_prob`, `dit_velocity_step0`, `latents`, `agg_out`, `vae_hidden`, `waveform_stream`, `waveform`) with a cosine of 0.9999 or more, a line `stop at step 61 (golden 61)`, an end-to-end line `62 steps (golden 62), stopped true`, and last `PARITY PASS`.
- **Every case.** Run `for c in emma_short emma_long george_short clone_george clone_lingguang; do $V/drive.sh parity-$c -- --parity "$G/$c" | tail -1; done`. Each run ends with `exit 0`, and each `stdout.txt` ends with `PARITY PASS`. A change to the engine, the prompt or the tokenizer needs all five; a clone change needs at least the two clone cases.
- **Failure shape.** A failing tensor prints `FAIL` in place of `ok`, the run ends with `PARITY FAIL`, and `exit.txt` holds 1. Cite the first `FAIL` line: later tensors fail because of it.
- **Audio.** `out.wav` in the evidence directory is the end-to-end run. For `emma_short`, `wavstat.json` shows 4.96 s.

## Gotchas

- `PARITY PASS` covers the teacher-forced tensors and the prompt. The end-to-end waveform line is printed for information and does not gate the result; the tool prints `drift expected after many steps`. Read the step count against the golden's yourself.
- `PARITY PASS` does not prove the `--say` path. Parity feeds the golden's noise and embeddings; `--say` draws its own noise and, for a clone, encodes the clip itself. Pair parity with a recipe from [speak.md](./speak.md) or [clone.md](./clone.md).
- `tokenizer_cases.json` and `STREAMING.md` in the goldens directory are not case directories.
- A case directory's `reference.wav` is the reference output, useful for listening against `out.wav`.
