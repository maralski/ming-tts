# Task modes

One flag group chooses how the model is directed: a plain read, a described voice, a free-form instruction, an emotion, speed, pitch and volume levels, or a Chinese dialect. A command may use the flags of one task only.

## Sub-features

- `task-plain` reads the text with no voice or style direction.
- `task-style` designs a voice from a description.
- `task-instruction` follows a free-form instruction.
- `task-emotion` speaks with one of seven emotions.
- `task-attributes` sets speed, pitch and volume to low, medium or high, alone or together.
- `task-dialect` speaks Cantonese or Chuanyu.
- `task-conflict` rejects the flags of two tasks, and values outside a flag's list.

## How to get to it (user POV)

- `ming-tts --say "text" --plain`
- `ming-tts --say "text" --style "a gravelly old sea captain"`
- `ming-tts --say "text" --instruction "speak softly, unhurried"`
- `ming-tts --say "text" --emotion happy|sad|angry|fearful|disgusted|surprised|neutral`
- `ming-tts --say "text" --speed low|medium|high`, and the same values for `--pitch` and `--volume`
- `ming-tts --say "text" --dialect cantonese|chuanyu`

## Driving it with drive.sh

Preconditions:

- `$V/doctor.sh` exits 0 and reports `whisper with small.en`.

- **Plain.** Run `$V/drive.sh task-plain -- --say "The garden stays open until sunset." --plain`. Exit 0, `stopped true`, and the transcript matches the text.
- **Voice design.** Run `$V/drive.sh task-style -- --say "Hoist the mainsail before the storm arrives." --style "a gravelly old sea captain"`. Exit 0, `stopped true`, about 3 s of audio, and the transcript reads `Hoist the mainsail before the storm arrives.`
- **Instruction.** Run `$V/drive.sh task-instruction -- --say "The garden stays open until sunset." --instruction "speak softly, unhurried"`. Exit 0, `stopped true`, and the transcript matches.
- **Emotion.** Run `$V/drive.sh task-emotion -- --say "I missed the last train home." --emotion sad`. Exit 0, `stopped true`, and the transcript matches.
- **Attributes.** Run `$V/drive.sh task-slow -- --say "The garden stays open until sunset." --speed low` and `$V/drive.sh task-fast -- --say "The garden stays open until sunset." --speed high`. Both transcripts match. Compare `seconds` in the two `wavstat.json` files: the slow run is expected to be longer. Run `$V/drive.sh task-attributes -- --say "The garden stays open until sunset." --speed low --pitch low --volume high` to prove the three combine in one command.
- **Dialect.** Run `$V/drive.sh task-dialect -- --say "今日天气好好，我哋去饮茶啦。" --dialect cantonese`. Exit 0, `stopped true`, about 3 s of audio with a peak well above 0. No transcript is made.
- **Conflict.** Run `$V/drive.sh task-conflict -- --say hi --plain --emotion sad`. Exit 2 with `bad input: give the flags of one task only` and no `out.wav`.
- **Bad value.** Run `$V/drive.sh task-bad-emotion -- --say hi --emotion bogus`. Exit 2 with `bad input: --emotion takes happy|sad|angry|fearful|disgusted|surprised|neutral, not bogus`.

## Gotchas

- The transcript proves the words. Whether the voice is gravelly, the read is sad or the volume is high needs a listener. The only attribute with a number is speed, through the audio length, and only against a run of the same text and seed.
- With no task flag the task is the built-in voice `US_Emma`, so a run that drops a flag still succeeds. Check `command.txt` for the flag before you cite a run.
- `--dialect` sends no target language by default, and a request with a dialect and a language is rejected. Do not add `--language` to a dialect run unless that rejection is what you are proving.
- An empty `--style ""` or `--instruction ""` is rejected as bad input.
- Bad values are checked after the checkpoint loads, so a `loaded` line on stdout comes before the error.
