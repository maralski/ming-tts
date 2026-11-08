// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A built-in Ming-3.0-tts voice, by its `ip` name in the model's voice catalog (`MingVoice.catalog`).
public struct MingVoice: Hashable, Sendable {
    public var ip: String
    public init(ip: String) { self.ip = ip }
    public static let emma = MingVoice(ip: "US_Emma")
    public static let george = MingVoice(ip: "UK_George")
    public static let charlotte = MingVoice(ip: "UK_Charlotte")
}

/// The language named on the prompt's `Target language:` line. Japanese, Korean, German, French
/// and Spanish text reaches the model as written, without normalization or a closing period.
public enum MingLanguage: String, CaseIterable, Sendable {
    case chinese = "Chinese", english = "English", japanese = "Japanese", korean = "Korean"
    case german = "German", french = "French", spanish = "Spanish"

    /// `tts_tasks._TARGET_LANGUAGE_NAMES`: a short code (`ja`) or a full name, any case.
    public init?(id: String) {
        let codes: [String: MingLanguage] = [
            "zh": .chinese, "zh-cn": .chinese, "en": .english, "ja": .japanese, "ko": .korean,
            "de": .german, "fr": .french, "es": .spanish, "yue": .chinese, "zh-yue": .chinese, "cantonese": .chinese,
        ]
        let id = id.trimmingCharacters(in: .whitespaces).lowercased()
        guard let language = codes[id] ?? Self.allCases.first(where: { $0.rawValue.lowercased() == id }) else { return nil }
        self = language
    }

    /// `tts_tasks.preserve_multilingual_text`: text the Chinese/English frontend must not touch.
    var preservesText: Bool { self != .chinese && self != .english }
}

/// Emotion control values; the raw values are the strings the checkpoint was trained on.
public enum MingEmotion: String, CaseIterable, Sendable {
    case happy = "高兴", sad = "悲伤", angry = "愤怒", fearful = "恐惧", disgusted = "厌恶", surprised = "惊讶", neutral = "中性"
}

/// A low, medium or high setting for speed, pitch or volume.
public enum MingLevel: CaseIterable, Sendable {
    case low, medium, high
    var speed: String { ["慢速", "中速", "快速"][index] }
    var pitchOrVolume: String { ["低", "中", "高"][index] }
    private var index: Int { Self.allCases.firstIndex(of: self)! }
}

public enum MingDialect: String, CaseIterable, Sendable {
    case cantonese = "cantonese", chuanyu = "chuanyu dialect"
}

/// One task mode of the reference `build_task_generation_kwargs`, with its controls.
public enum MingTask: Sendable {
    /// TTS, or MULTILINGUAL when the request's language is Japanese, Korean, German, French or Spanish.
    case plain
    /// IP: a built-in voice.
    case builtInVoice(MingVoice)
    /// VOICE_DESIGN: a description of the voice.
    case voiceDesign(String)
    /// FREEFORM: one natural-language instruction for voice and delivery, sent as written.
    case freeform(String)
    case emotion(MingEmotion)
    /// BASIC: at least one of the three.
    case attributes(speed: MingLevel?, pitch: MingLevel?, volume: MingLevel?)
    /// DIALECT: Chinese text read in a dialect; takes no language.
    case dialect(MingDialect)

    /// The reference's STYLE runtime task (VOICE_DESIGN and FREEFORM).
    var isStyle: Bool {
        switch self {
        case .voiceDesign, .freeform: true
        default: false
        }
    }
}

/// An encoded reference voice: the acoustic prompt a request clones. Encode a clip once with
/// `MingTTSModel.encodeReference`, persist it, and reuse it for every request.
public struct MingReferenceVoice: Sendable, Codable, Equatable {
    public static let latentDim = 64
    public static let speakerEmbeddingDim = 192
    /// What the clip says, as the model should read it.
    public let transcript: String
    /// VAE latents of the clip, `latentRows` x 64 row-major, after the reference's front padding.
    public let latents: [Float]
    /// The 192-dim CAM++ speaker embedding; the model's `spk_head` projects it when the prompt is built.
    public let speakerEmbedding: [Float]

    public init(transcript: String, latents: [Float], speakerEmbedding: [Float]) throws {
        let transcript = pythonStrip(transcript)
        guard !transcript.isEmpty else { throw MingTTSError.badInput("a reference voice needs its transcript") }
        guard !latents.isEmpty, latents.count % Self.latentDim == 0 else { throw MingTTSError.badInput("reference latents are not rows of \(Self.latentDim)") }
        guard speakerEmbedding.count == Self.speakerEmbeddingDim else { throw MingTTSError.badInput("speaker embedding must have \(Self.speakerEmbeddingDim) values") }
        self.transcript = transcript
        self.latents = latents
        self.speakerEmbedding = speakerEmbedding
    }

    public var latentRows: Int { latents.count / Self.latentDim }
}

/// What the prompt conditions the voice on. `noSpeaker` has no speaker row; `zeroSpeaker` is the
/// reference `use_zero_spk_emb`, one speaker row of zeros; `clone` splices the reference clip's
/// transcript and latents into the prompt and, unless `speakerEmbedding` is false, its projected
/// speaker embedding into the speaker row (the reference `use_spk_emb`).
public enum MingConditioning: Sendable {
    case noSpeaker
    case zeroSpeaker
    case clone(MingReferenceVoice, speakerEmbedding: Bool = true)

    /// Whether the prompt has a speaker row at all.
    var hasSpeakerRow: Bool {
        switch self {
        case .noSpeaker: false
        case .zeroSpeaker: true
        case .clone(_, let embedding): embedding
        }
    }

    var reference: MingReferenceVoice? {
        if case .clone(let voice, _) = self { voice } else { nil }
    }
}

/// Everything one synthesis needs. The initializer rejects what the reference rejects.
public struct MingRequest: Sendable {
    /// Stripped of surrounding whitespace, as the reference does before preparing it.
    public let text: String
    public let task: MingTask
    public let language: MingLanguage?
    public private(set) var conditioning: MingConditioning

    /// `conditioning` has no default on purpose: the right value depends on the task (plain TTS with
    /// a zero speaker row can run to the decode limit, built-in voices expect one), so the caller decides.
    public init(text: String, task: MingTask, language: MingLanguage? = nil, conditioning: MingConditioning) throws {
        let text = pythonStrip(text)
        guard !text.isEmpty else { throw MingTTSError.badInput("text must not be empty") }
        switch task {
        case .voiceDesign(let s) where pythonStrip(s).isEmpty:
            throw MingTTSError.badInput("voice design needs a description")
        case .freeform(let s) where pythonStrip(s).isEmpty:
            throw MingTTSError.badInput("freeform needs a non-empty instruction")
        case .attributes(nil, nil, nil):
            throw MingTTSError.badInput("attributes need a speed, pitch or volume")
        case .dialect where language != nil:
            throw MingTTSError.badInput("a dialect takes no language")
        default: break
        }
        self.text = text
        self.task = task
        self.language = language
        self.conditioning = conditioning
    }

    /// The same request conditioned differently (a STYLE task clones its first fragment for the rest).
    func conditioned(_ conditioning: MingConditioning) -> MingRequest {
        var copy = self
        copy.conditioning = conditioning
        return copy
    }
}
