// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Token ids for one generation and the rows whose embeddings the engine replaces with continuous
/// features: the speaker row (zeros, or the projected speaker embedding) and, for a clone, one row
/// per reference latent from `latentPosition` on. `conditioning` says what fills them.
public struct MingPrompt: Sendable {
    public var ids: [Int32]
    public var speakerPositions: [Int]
    /// The first `<audioPatch>` row; nil without an acoustic prompt.
    public var latentPosition: Int?
    public var conditioning: MingConditioning

    /// The reference latents spliced in, if any.
    var reference: MingReferenceVoice? { conditioning.reference }
}

/// `prompt_contract.build_input_ids` for one prepared fragment of `request`, with the prompt and
/// instruction that `tts_tasks.build_task_generation_kwargs` resolves for its task. A clone puts
/// the reference transcript before the text and one `<audioPatch>` per reference latent after
/// `<audio>`. Each piece is encoded on its own and concatenated, as the reference does.
func buildPrompt(fragment: String, request: MingRequest, tokenizer: MingTokenizer) -> MingPrompt {
    let bgmMarkers = ["Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: "]
    let conditioning = request.conditioning
    let speakerBlock = conditioning.hasSpeakerRow
        ? ["  speaker_1:", "<|vision_start|>", "<|vision_pad|>", "<|vision_end|>\n"] : []
    let reference = conditioning.reference
    let pieces = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        "<|im_start|>user\n",
        "Please generate speech based on the following description.\n"
            + (request.language.map { "Target language: \($0.rawValue).\n" } ?? ""),
    ] + speakerBlock + [
        bgmMarkers.allSatisfy { fragment.contains($0) } ? "" : " Text input:\n",
        reference?.transcript ?? "",
        fragment,
        "<|im_end|>\n", "<|im_start|>assistant\n",
    ] + (instruction(for: request.task).map { [$0, "<|im_end|>"] } ?? []) + [
        "<audio>",
    ]
    var ids = pieces.flatMap(tokenizer.encode)
    let speakerStart = tokenizer.addedTokenID("<|vision_start|>")!
    let speakerPositions = ids.indices.filter { ids[$0] == speakerStart }.map { $0 + 1 }
    precondition(speakerPositions.count == (conditioning.hasSpeakerRow ? 1 : 0), "text must not contain <|vision_start|>")
    var latentPosition: Int?
    if let reference {
        latentPosition = ids.count
        ids += [Int32](repeating: tokenizer.addedTokenID("<audioPatch>")!, count: reference.latentRows)
    }
    return MingPrompt(ids: ids, speakerPositions: speakerPositions, latentPosition: latentPosition, conditioning: conditioning)
}

/// The assistant-side caption; nil for plain TTS.
func instruction(for task: MingTask) -> String? {
    switch task {
    case .plain: nil
    case .freeform(let text): pythonStrip(text)
    case .builtInVoice(let voice): structuredInstruction(.zh, ip: voice.ip)
    case .voiceDesign(let style): structuredInstruction(.zh, [.style: style])
    case .emotion(let emotion): structuredInstruction(.zh, [.emotion: emotion.rawValue])
    case .attributes(let speed, let pitch, let volume):
        structuredInstruction(.zh, [.speed: speed?.speed, .pitch: pitch?.pitchOrVolume, .volume: volume?.pitchOrVolume].compactMapValues { $0 })
    case .dialect(let dialect): structuredInstruction(.en, [.dialect: dialect.rawValue])
    }
}

enum InstructionSchema { case zh, en }

/// The controllable fields of `build_structured_instruction`, in caption order.
enum InstructionField: CaseIterable {
    case dialect, style, speed, pitch, volume, emotion
}

/// `tts_tasks.build_structured_instruction` serialized as Python's `json.dumps(ensure_ascii=False)`.
/// The key order is part of the trained protocol.
func structuredInstruction(_ schema: InstructionSchema, _ values: [InstructionField: String] = [:], ip: String? = nil) -> String {
    let keys: [String], ipKey: String
    switch schema {
    case .zh: (keys, ipKey) = (["序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感"], "IP")
    case .en: (keys, ipKey) = (["id", "speaker", "dialect", "style", "speed", "pitch", "volume", "emotion"], "ip")
    }
    let fieldValues = [JSON.int(1), .string("speaker_1")] + InstructionField.allCases.map { JSON(values[$0]) }
    let bgm = JSON.object(["Genre", "Mood", "Instrument", "Theme", "ENV", "SNR"].map { ($0, .null) })
    let item = JSON.object(Array(zip(keys, fieldValues)) + [("BGM", bgm), (ipKey, JSON(ip))])
    return JSON.object([("audio_sequence", .array([item]))]).python
}

private indirect enum JSON {
    case null, int(Int), string(String), array([JSON]), object([(String, JSON)])

    init(_ value: String?) { self = value.map(JSON.string) ?? .null }

    /// Python `json.dumps` with its default separators.
    var python: String {
        switch self {
        case .null: "null"
        case .int(let n): String(n)
        case .string(let s): "\"" + jsonEscaped(s) + "\""
        case .array(let items): "[" + items.map(\.python).joined(separator: ", ") + "]"
        case .object(let pairs): "{" + pairs.map { "\"" + jsonEscaped($0.0) + "\": " + $0.1.python }.joined(separator: ", ") + "}"
        }
    }
}

private func jsonEscaped(_ s: String) -> String {
    var out = ""
    for scalar in s.unicodeScalars {
        switch scalar {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        case "\u{08}": out += "\\b"
        case "\u{0C}": out += "\\f"
        case _ where scalar.value < 0x20: out += String(format: "\\u%04x", scalar.value)
        default: out.unicodeScalars.append(scalar)
        }
    }
    return out
}
