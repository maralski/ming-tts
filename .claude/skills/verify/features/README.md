# MingTTS verification map

This directory is the maintained source for verifying the user-facing behavior of the `ming-tts` command-line tool. Read this index before driving the tool, then use the matching feature file as the recipe.

## Baseline preconditions

- Run every command from the repository root.
- `swift build -c release` has finished since the last source change.
- `.claude/skills/verify/scripts/doctor.sh` exits 0. Its warnings name the features that cannot run: no `campplus.onnx` blocks cloning, no goldens block parity, no whisper removes transcripts.
- `V` stands for `.claude/skills/verify/scripts`.
- `G` stands for the goldens directory that the doctor prints (`MING_GOLDENS`, else `../ming-ref/goldens`).

## Driving conventions

- Run the tool only through `$V/drive.sh <name> -- <ming-tts arguments>`. It files the command, output, exit code, audio, audio facts and transcript.
- Treat every command as literal. Keep quoted text and flags unchanged.
- Give each run a distinct `<name>`; it labels the evidence directory.
- The tool keeps no state between runs, so there is no baseline to restore.
- Do not overlap runs when a timing is part of the proof.

## Proof and skip reporting

- CLI proof is the evidence directory: `command.txt`, `stdout.txt`, `stderr.txt`, `exit.txt`.
- Speech proof adds `wavstat.json` and, for English, `transcript.txt`. Require `stopped true` on every `fragment:` line, a peak well above 0, and transcript words that match the input.
- A transcript does not prove how the audio sounds. Say so, and send `out.wav` to the user when timbre, emotion or likeness is the claim.
- Record the feature ID and the entry point with every artifact you cite.
- Report an entry point you could not run with the command you tried and the precondition that was unmet. Do not report it as verified through another path.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with drive.sh` starts with `Preconditions:` and uses labeled bullets that pair each user action with an exact command and observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only user paths, stable handles, required state, commands, and observable proof.

## Features

- [Speak text](./speak.md) covers `--say` with the default and named built-in voices, English normalization, long text in fragments, sound tags, languages, seeds, checkpoint discovery and bad arguments.
- [Task modes](./task-modes.md) covers `--plain`, `--style`, `--instruction`, `--emotion`, `--speed`, `--pitch`, `--volume`, `--dialect`, and their conflicts.
- [Clone a voice](./clone.md) covers `--clone` with `--transcript`, `--anchor`, and `--speaker`.
- [Golden parity](./parity.md) covers `--parity` against tensors captured from the Python reference.
- [Benchmark](./bench.md) covers `--bench` and comparing two builds.
