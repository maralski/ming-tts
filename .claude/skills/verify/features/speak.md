# Speak text

`ming-tts --say "text"` turns text into speech in a built-in voice and writes one WAV. English text is normalized first (numbers, money, dates, times), long text is spoken in fragments, and bracketed sound tags are performed instead of read.

## Sub-features

- `speak-default` speaks in the default voice `US_Emma` and writes `out.wav`.
- `speak-voice` speaks in a named built-in voice with `--voice`.
- `speak-normalize` reads English numbers, money, dates and times as words.
- `speak-fragments` splits long text and writes `out.fragN.wav` beside `out.wav`.
- `speak-tags` performs a sound tag such as `[Sigh]` in place.
- `speak-language` sets the target language with `--language`.
- `speak-seed` makes output repeatable with `--seed`.
- `speak-weights` finds the checkpoint from `--weights`, `MING_WEIGHTS` or the Hugging Face cache, and names the missing files of a bad folder.
- `speak-errors` rejects an unknown argument or a flag with no value.

## How to get to it (user POV)

- Run `.build/release/ming-tts --say "text"`; the WAV goes to `out.wav` in the working directory.
- Add `--out file.wav` to choose the file.
- Add `--voice UK_George` to choose a built-in voice.
- Add `--language en|ja|ko|de|fr|es|zh|zh-Hant|none`, or a language name such as `Japanese`.
- Add `--seed n`.
- Pass the checkpoint with `--weights dir`, or set `MING_WEIGHTS`, or rely on the Hugging Face cache.
- Run `.build/release/ming-tts` with no arguments for the usage line.

## Driving it with drive.sh

Preconditions:

- `$V/doctor.sh` exits 0 and reports `whisper with small.en`.

- **Default voice.** Speak one sentence. Run `$V/drive.sh speak-default -- --say "The garden stays open until sunset."`. Exit 0; stdout has `loaded`, one `fragment: N steps, stopped true` line, a totals line with `peak` near 0.3, and `wrote .../out.wav`. `wavstat.json` shows float, 1 channel, 44100 Hz and about 2.6 s. `transcript.txt` reads `The garden stays open until sunset.`
- **Named voice.** Run `$V/drive.sh speak-voice -- --say "The garden stays open until sunset." --voice UK_George`. Same lines as above. The transcript matches. The voice itself needs a listener.
- **Normalization.** Run `$V/drive.sh speak-normalize -- --say "Meet at 3:30pm on Nov 6. It costs \$12.50."`. The transcript carries the time, the date and the amount, for example `Meet at 3.30 p.m. on 6 November. It costs $12.50.` Whisper writes them back as digits, so judge by the spoken words: a transcript such as `Meet at three colon thirty` is a failure.
- **Fragments.** Run `$V/drive.sh speak-fragments -- --say "The garden will remain open until sunset, provided the weather stays clear. The volunteers who tend the roses will stay to answer questions from visitors. After dark the gates are locked and the lamps along the east path are switched on for the night watchman. Please take all of your belongings with you when you leave."`. Stdout has four `fragment:` lines, all `stopped true`, and `wrote` lines for `out.wav` and `out.frag1.wav` to `out.frag4.wav`. The transcript holds all four sentences in order. In `wavstat.json`, the seconds of the fragments add up to the seconds of `out.wav`.
- **Sound tag.** Run `$V/drive.sh speak-tags -- --say "Well [Sigh] that is that."`. `stopped true`, and the transcript reads `Well, that is that.` with no word `sigh`. Whether a sigh is audible needs a listener.
- **Language.** Run `$V/drive.sh speak-language -- --say "こんにちは、今日はいい天気ですね。" --language ja`. Exit 0, `stopped true`, about 2.5 s of audio. No transcript is made for a non-English run. Run `$V/drive.sh speak-language-bad -- --say hi --language xx`. Exit 2 with `bad input: unknown language xx` and no `out.wav`.
- **Seed.** Run `$V/drive.sh speak-seed-a -- --say "Hello there." --seed 7`, then `$V/drive.sh speak-seed-b -- --say "Hello there." --seed 7`. `cmp` on the two `out.wav` files reports no difference. A third run with `--seed 8` differs.
- **Checkpoint.** Run `$V/drive.sh speak-weights-bad -- --say hi --weights /nonexistent`. Exit 2 with `bad checkpoint: /nonexistent lacks model.safetensors, tokenizer.json` on stderr and no `out.wav`. To prove `MING_WEIGHTS`, run `MING_WEIGHTS=<checkpoint dir> $V/drive.sh speak-weights-env -- --say "Hello there."` and check that the `loaded` line names that directory.
- **Bad arguments.** Run `$V/drive.sh speak-unknown -- --say hi --bogus`. Exit 2 with `bad input: unknown argument --bogus`. Run `$V/drive.sh speak-novalue -- --say`. Exit 2 with `--say needs a value`.

## Gotchas

- An unknown `--voice` name is not rejected. `--voice No_Such_Voice` exits 0 and speaks, so exit 0 does not prove the voice exists. Take names from `MingVoice.catalog`.
- `drive.sh` adds `--weights` whenever it finds a checkpoint, so it never exercises the Hugging Face cache search. To prove the cache path, run the binary directly with `MING_WEIGHTS` unset and no `--weights`, and capture the output yourself.
- There is no `--help`; it is an unknown argument and exits 2.
- Whisper respells numbers, punctuation and capitals. Compare words.
- A `[Laughter]` tag can run to the step limit and print `stopped false`. That is known model behavior, not a regression by itself; compare against the same line and seed before your change.
- Chinese text is not normalized, by design.
