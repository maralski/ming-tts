# Benchmark

`ming-tts --bench` speaks a fixed paragraph several times and prints, for each run, the step rate, the real-time factor, the time to first audio and the GPU time of each stage. `scripts/ab_bench.py` compares two builds.

## Sub-features

- `bench-run` prints one `run N:` line and one `per step ms:` line for each run.
- `bench-options` takes `--text`, `--repeat n`, and the task and clone flags of `--say`.
- `bench-ab` compares two binaries in interleaved rounds and prints `B/A steps/s`.

## How to get to it (user POV)

- `ming-tts --bench`
- `ming-tts --bench --text "other text" --repeat 5`
- `scripts/gpulock.py -- .build/release/ming-tts --bench`, to keep two measurements from overlapping.
- `scripts/ab_bench.py BASE_BINARY CANDIDATE_BINARY --rounds 5`, with `MING_WEIGHTS` set or `--weights dir`.

## Driving it with drive.sh

Preconditions:

- `$V/doctor.sh` exits 0 with no `ming-tts already running` warning.
- Nothing else heavy is using the GPU. Do not run other drives while a benchmark runs.

- **Run.** Run `$V/drive.sh bench-run -- --bench --repeat 3`. Exit 0. Stdout has `bench: 174 prompt tokens, 3 runs, ...`, then for each run a line such as `run 1: 110 steps, 8.80 s audio, wall 1.74 s, RTF 0.198, 64.5 steps/s, first audio 65 ms, prefill 26.0 ms` and a line `per step ms: wall 15.50  gpu 15.01  [llm 5.24  flow 8.93  agg 0.84  vae 0.01]`. No WAV is written. Those figures are from an M-series laptop in November 2026; take your own baseline on the machine you measure on.
- **Options.** Run `$V/drive.sh bench-options -- --bench --text "The garden stays open until sunset." --repeat 2`. The `bench:` line reports fewer prompt tokens and `2 runs`, and two `run` lines follow.
- **Compare two builds.** Build the base commit into a second directory, for example `git worktree add "$TMPDIR/ming-base" <base commit>` and `swift build -c release` there. Then run `MING_WEIGHTS=<checkpoint dir> scripts/ab_bench.py "$TMPDIR/ming-base/.build/release/ming-tts" .build/release/ming-tts --rounds 5 | tee .audit/verify/ab-<name>.txt`. It prints an `A` line and a `B` line of medians, a `range` line under each, and `B/A steps/s = N`. Remove the worktree afterwards with `git worktree remove "$TMPDIR/ming-base"`.

## Gotchas

- Run 0 is a cold run: its first-audio and prefill times are about twice those of later runs. Compare run 1 onward. `ab_bench.py` already drops run 0.
- A speed claim needs the A/B script, not two separate `--bench` runs. Machine state moves the numbers more than most changes do; report the median with its range, and call a difference inside the ranges no change.
- `drive.sh` holds the GPU lock for `--bench` only. A `--say` run in another terminal still competes for the GPU and skews the numbers.
- The step count and audio length are fixed by the text and seed. If `110 steps, 8.80 s audio` changes for the default text, the change altered the output, not only the speed. Check [parity.md](./parity.md).
- `ab_bench.py` needs `MING_WEIGHTS` or `--weights`; it does not use the helper's search order.
- `--gemv-bench`, `--qwen-bench` and `--simd-layout` are developer microbenchmarks, not user features, and are not mapped here.
