# Clone a voice

`ming-tts --say "text" --clone ref.wav --transcript "what the clip says"` speaks new text in the voice of a reference clip. `--anchor` does the same for a built-in voice: the tool first has that voice read a fixed line, then clones that clip, so the timbre holds from one utterance to the next.

## Sub-features

- `clone-clip` clones the voice of a reference clip, given its transcript.
- `clone-anchor` clones a built-in voice from the tool's own anchor clip.
- `clone-speaker` turns the speaker embedding off with `--speaker none`, with or without a clone.
- `clone-errors` rejects `--clone` with no `--transcript`, and `--anchor` with no `--voice` or with `--clone`.

## How to get to it (user POV)

- `ming-tts --say "text" --plain --clone ref.wav --transcript "what the clip says"`
- `ming-tts --say "text" --voice UK_George --anchor`
- Add `--speaker none` to a `--clone` run to clone from the clip's latents alone.
- Add `--seed n`; with `--clone` it also seeds how the clip is encoded.

## Driving it with drive.sh

Preconditions:

- `$V/doctor.sh` exits 0 and reports `campplus.onnx present` and `whisper with small.en`.
- A reference clip whose words you know exactly. Make one with the tool, so the recipe needs no outside file: run `$V/drive.sh clone-ref -- --say "Markets rallied this morning after the central bank held interest rates steady." --voice UK_George`, and use that run's `out.wav` as `REF`. Its transcript must read back those words.

- **Clone a clip.** Run `$V/drive.sh clone-clip -- --say "This sentence was never in the reference clip." --plain --clone "$REF" --transcript "Markets rallied this morning after the central bank held interest rates steady."`. Exit 0; stdout has `encoded the reference clip in N ms`, then `fragment: N steps, stopped true` with about 30 steps, and a totals line with about 2.4 s of audio and a peak near 0.35. The transcript reads `This sentence was never in the reference clip.` and holds none of the reference's words.
- **Anchor.** Run `$V/drive.sh clone-anchor -- --say "This sentence was never in the reference clip." --voice UK_George --anchor`. Exit 0, `stopped true`, about 3 s of audio, and the transcript matches. The run takes a few seconds longer, because it speaks the anchor line first.
- **No speaker embedding, no clone.** Run `$V/drive.sh speaker-none -- --say "The garden stays open until sunset." --speaker none`. Exit 0, `stopped true`, about 2.7 s of audio, and the transcript matches. Run `$V/drive.sh speaker-bad -- --say hi --speaker maybe`. Exit 2 with `bad input: --speaker takes zero|none`.
- **No speaker embedding, with a clone.** Run the clone command again as `clone-nospeaker` with `--speaker none` added, on the untrimmed `REF`. Exit 0, `stopped true` with about 36 steps, about 2.9 s of audio, a peak near 0.35, and the transcript matches. This run is the regression check for trimming: before `--clone` trimmed the clip, it gave silence (`5 steps`, `peak 0.000`) at every seed.
- **Silent clip.** Run `--clone` on a clip of pure silence. Exit 2 with `bad input: reference audio is empty` and no `out.wav`.
- **Missing transcript.** Run `$V/drive.sh clone-no-transcript -- --say hi --clone "$REF"`. Exit 2 with `bad input: --clone needs --transcript` and no `out.wav`.
- **Anchor misuse.** Run `$V/drive.sh clone-anchor-plain -- --say hi --plain --anchor`. Exit 2 with `bad input: --anchor needs --voice and no --clone`.
- **Proof of likeness.** The transcript proves the words, not the voice. Send the reference clip and the clone's `out.wav` to the user together and say the likeness was not judged by ear.

## Gotchas

- The transcript must be the clip's exact words. A wrong transcript still exits 0 with `stopped true`, but the audio is about 0.4 s of silence (`5 steps`, `peak 0.000`) or 30 s of babble (`stopped false`). Treat either as "the transcript does not match the clip" before you suspect the code.
- In a golden case directory, `reference.wav` is the Python reference's output for that case, not the clone prompt. The prompt clip is `prompt_wave_44k.f32` (raw 32-bit float, mono, 44,100 Hz), and its words are `prompt_text` in `manifest.json`. Convert it with `ffmpeg -f f32le -ar 44100 -ac 1 -i prompt_wave_44k.f32 -c:a pcm_f32le prompt.wav`.
- Exit 0 and `stopped true` are not enough for this feature. Always read `peak` and the transcript.
- `--clone` and `--anchor` trim silence from both ends of the clip, so a trailing pause in `REF` does not matter through the tool. The library call `encodeReference(url:transcript:)` does not trim. A clone that stops at `5 steps` with `peak 0.000` on a correct transcript means the trim is not taking effect; check `trailing_silence_seconds` with `$V/wavstat.py "$REF"` and rerun with a hand-trimmed clip to confirm.
- Keep the clip under 15 s. A longer one leaves too little room to speak.
- `--anchor` works only with a built-in voice, and the default task counts: `--say hi --anchor` with no task flag uses `US_Emma`.
