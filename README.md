# MingTTS

MingTTS runs [Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B) by inclusionAI on a Mac. It is a Swift and Metal port of the model's inference path, with no Python at run time. The package has two products:

- `MingTTS`, a library that turns text into 44.1 kHz mono audio, streamed fragment by fragment.
- `ming-tts`, a command-line tool that writes speech to a WAV file and runs benchmarks.

The port covers the upstream task modes: built-in voices, plain text to speech, voice design, free-form instructions, emotions, speed, pitch and volume, Chinese dialects, and voice cloning from a reference clip. English text is normalized in Swift (numbers, currencies, dates, times, units and abbreviations). Chinese text normalization is not ported.

## Requirements

- A Mac with Apple silicon.
- macOS 14 or later.
- Xcode 16 or later (Swift 6 tools).

## Get the weights

The weights are not in this repository. Download the checkpoint from Hugging Face:

```sh
pip install -U huggingface_hub
hf download inclusionAI/Ming-3.0-tts-0.5B --local-dir ~/models/Ming-3.0-tts-0.5B
```

The directory must hold `model.safetensors`, `config.json`, `tokenizer.json` and `campplus.onnx`. The weights are under inclusionAI's license, not this repository's. See [NOTICE](NOTICE).

## Build

```sh
swift build -c release
```

## Use the command-line tool

Pass the checkpoint directory with `--weights`, or set `MING_WEIGHTS`:

```sh
export MING_WEIGHTS=~/models/Ming-3.0-tts-0.5B
.build/release/ming-tts --say "The garden stays open until sunset." --out garden.wav
```

Choose a task with one of these options. The default is the built-in voice `US_Emma`.

| Option | Task |
| --- | --- |
| `--voice UK_George` | A built-in voice. |
| `--plain` | Plain text to speech. |
| `--style "a gravelly old sea captain"` | Voice design from a description. |
| `--instruction "speak softly, unhurried"` | A free-form instruction. |
| `--emotion happy` | An emotion: happy, sad, angry, fearful, disgusted, surprised or neutral. |
| `--speed low`, `--pitch low`, `--volume high` | Speed, pitch and volume: low, medium or high. |
| `--dialect cantonese` | A Chinese dialect: cantonese or chuanyu. |
| `--clone ref.wav --transcript "..."` | Clone the voice in a reference clip. |

Other options are `--language en|ja|...|none`, `--speaker zero|none` and `--seed n`. To measure speed, run `ming-tts --bench`.

## Use the library

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/maralski/ming-tts.git", from: "0.1.0")
```

Then load the model once and stream audio:

```swift
import MingTTS

let model = try MingTTSModel(directory: URL(fileURLWithPath: "/path/to/Ming-3.0-tts-0.5B"))
let request = try MingRequest(
    text: "Meet at 3:30pm on Nov 6.",
    task: .builtInVoice(.emma),
    language: .english,
    conditioning: .zeroSpeaker)

for try await chunk in model.stream(request) {
    // chunk is [Float] PCM at MingTTSModel.sampleRate (44,100 Hz), mono.
}
```

To clone a voice, encode a reference clip and pass it as the conditioning:

```swift
let voice = try model.encodeReference(url: clipURL, transcript: "What the clip says.")
let cloned = try MingRequest(text: "Hello.", task: .plain, language: .english,
                             conditioning: .clone(voice))
```

`MingReferenceVoice` is `Codable`, so you can store an encoded voice and reuse it.

## Test

```sh
MING_WEIGHTS=~/models/Ming-3.0-tts-0.5B swift test -c release -Xswiftc -enable-testing
```

The speed tests assume an optimized build. In a debug build, `textTokenizerIsFastOnABulletin` fails its 5 ms bound.

Tests that need files outside the repository are skipped when those files are not set:

- `MING_WEIGHTS` enables the tokenizer, prompt and clone tests.
- `MING_GOLDENS` enables the parity, voice and clone golden tests. The goldens are tensors captured from the Python reference and are not published.
- `MING_CLONE_TIMING_WAV` and `MING_VOICE_TIMING_WAV` enable two timing tests.

The fixtures in `Tests/MingTTSTests/Fixtures` come from the upstream Python frontend and the Hugging Face tokenizer. The `make_*_fixtures.py` scripts next to them regenerate the fixtures; each script's docstring names the Python environment it needs. The input text is synthetic.

## License

MingTTS is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
