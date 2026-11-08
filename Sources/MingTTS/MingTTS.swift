// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import Foundation

public struct SynthesisOptions: Sendable {
    public var cfg: Float = 2.0
    public var flowSteps: Int = 10
    public var maxSteps: Int = 1000
    public var seed: UInt64 = 0
    /// Longest text fragment synthesised as one utterance, in the reference frontend's semantic
    /// length units (the reference default is 24).
    public var fragmentLength: Int = 24
    public init() {}
}

/// Ming-3.0-tts-0.5B on the GPU. Audio is 44.1 kHz mono Float32, produced while the model generates.
public final class MingTTSModel: @unchecked Sendable {
    public static let sampleRate = 44_100

    let engine: MingEngine
    let tokenizer: MingTokenizer
    private let directory: URL
    private var speakerNetCache: CAMPlus?
    private let speakerNetLock = NSLock()

    /// `directory` holds the Hugging Face checkpoint: model.safetensors, config.json, tokenizer.json
    /// and, for cloning, campplus.onnx.
    public init(directory: URL) throws {
        self.directory = directory
        engine = try MingEngine(directory: directory)
        tokenizer = try MingTokenizer(tokenizerJSON: directory.appendingPathComponent("tokenizer.json"))
    }

    // MARK: synthesis

    /// Streams PCM chunks for `request` as they are decoded. The first chunk arrives after a few generation steps.
    public func stream(_ request: MingRequest, options: SynthesisOptions = .init()) -> AsyncThrowingStream<[Float], Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) {
                do {
                    _ = try self.synthesize(request, options: options) { _, chunk in
                        try Task.checkCancellation()
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Synthesises every fragment of `request` in order, delivering PCM chunks with their fragment
    /// index as the GPU finishes them. A STYLE task (voice design, freeform) with several fragments
    /// clones its first fragment's audio for the rest, as the reference does, so the voice holds.
    public func synthesize(_ request: MingRequest, options: SynthesisOptions = .init(), onChunk: (Int, [Float]) throws -> Void) throws -> [GenerationReport] {
        let fragments = try prepareFragments(request, maxLength: options.fragmentLength)
        var request = request
        var reports: [GenerationReport] = []
        for (index, fragment) in fragments.enumerated() {
            let prompt = buildPrompt(fragment: fragment, request: request, tokenizer: tokenizer)
            let becomesReference = request.task.isStyle && index == 0 && fragments.count > 1
            var pcm: [Float] = []
            reports.append(try synthesize(prompt: prompt, options: options) { chunk in
                if becomesReference { pcm += chunk }
                try onChunk(index, chunk)
            })
            if becomesReference {
                let voice = try encodeReference(samples: trimTrailingSilence(pcm, sampleRate: Self.sampleRate), sampleRate: Self.sampleRate,
                                                transcript: fragment, seed: options.seed)
                request = request.conditioned(.clone(voice))
            }
        }
        return reports
    }

    /// The fragments `request` is spoken in, in order.
    public func fragments(for request: MingRequest, fragmentLength: Int = SynthesisOptions().fragmentLength) throws -> [String] {
        try prepareFragments(request, maxLength: fragmentLength)
    }

    /// The prompt for `fragment` sent as given, without text preparation or splitting.
    public func prompt(fragment: String, of request: MingRequest) -> MingPrompt {
        buildPrompt(fragment: fragment, request: request, tokenizer: tokenizer)
    }

    /// Synthesises one prompt synchronously, delivering PCM chunks as the GPU finishes them.
    /// `splitStages` runs each stage in its own command buffer so the report carries per-stage GPU time.
    public func synthesize(prompt: MingPrompt, options: SynthesisOptions = .init(), splitStages: Bool = false, onChunk: ([Float]) throws -> Void) throws -> GenerationReport {
        let embeds = try engine.promptEmbeddings(prompt: prompt)
        var o = MingEngine.Options()
        o.cfg = options.cfg
        o.maxSteps = options.maxSteps
        o.seed = options.seed
        o.initialHistory = prompt.reference.map(Self.initialHistory)
        o.splitStages = splitStages
        let outcome = try engine.generate(embeds: embeds, options: o, onChunk: onChunk)
        return GenerationReport(outcome: outcome)
    }

    /// The flow head's latent history at step 0: the reference's last latent (`history_patch_size` 1).
    static func initialHistory(_ voice: MingReferenceVoice) -> [Float] {
        Array(voice.latents.suffix(MingReferenceVoice.latentDim))
    }

    // MARK: reference voices

    /// What a built-in voice reads for its anchor clip.
    public static let anchorText = "Good morning. Here is the news, read slowly and clearly, with a calm and steady voice from start to finish."

    /// A reference clip of built-in `voice` to clone in every request in that voice, so its timbre holds from one
    /// utterance to the next, which the voice's prompt alone does not do. The clip is `anchorText` read with seed 0 and
    /// trimmed of silence at both ends: a clip ending in silence makes most clones stop after a few steps. The same voice
    /// gives the same clip, so persist it and reuse it.
    public func anchor(_ voice: MingVoice, language: MingLanguage = .english) throws -> MingReferenceVoice {
        let request = try MingRequest(text: Self.anchorText, task: .builtInVoice(voice), language: language, conditioning: .zeroSpeaker)
        var pcm: [Float] = []
        _ = try synthesize(request) { _, chunk in pcm += chunk }
        let clip = trimSilence(pcm, sampleRate: Self.sampleRate)
        let seconds = Double(clip.count) / Double(Self.sampleRate)
        guard seconds > 3 else { throw MingTTSError.badInput("\(voice.ip) read the anchor text in \(String(format: "%.1f", seconds)) s, too short to clone") }
        return try encodeReference(samples: clip, sampleRate: Self.sampleRate, transcript: Self.anchorText)
    }

    /// Encodes a reference clip for cloning: its CAM++ speaker embedding from a 16 kHz resample and
    /// its VAE latents from a 44.1 kHz resample, front-padded to whole 3528-sample patches as the
    /// reference pads. `seed` draws the posterior sample. Persist the result and reuse it. Clips over
    /// about 15 s leave the 512-position cache too little room to generate; the reference has the same limit.
    public func encodeReference(samples: [Float], sampleRate: Int, transcript: String, seed: UInt64 = 0) throws -> MingReferenceVoice {
        try encodeReference(samples: samples, sampleRate: sampleRate, transcript: transcript, noise: nil, seed: seed)
    }

    /// Decodes `url` with AVFoundation (any format it reads), mixes to mono, and encodes it.
    public func encodeReference(url: URL, transcript: String, seed: UInt64 = 0) throws -> MingReferenceVoice {
        let (samples, rate) = try Self.readMono(url)
        return try encodeReference(samples: samples, sampleRate: rate, transcript: transcript, seed: seed)
    }

    /// `noise` replaces the seeded posterior draw (parity tests inject the reference's).
    func encodeReference(samples: [Float], sampleRate: Int, transcript: String, noise: [Float]?, seed: UInt64) throws -> MingReferenceVoice {
        guard !samples.isEmpty else { throw MingTTSError.badInput("reference audio is empty") }
        guard sampleRate > 0 else { throw MingTTSError.badInput("sample rate must be positive") }
        let wave16k = resample(samples, from: sampleRate, to: 16000)
        let fbank = kaldiFbank(wave16k)
        guard fbank.frames > 0 else { throw MingTTSError.badInput("reference audio is shorter than one 25 ms frame") }
        let embedding = try speakerNet().embed(fbank: fbank.features, frames: fbank.frames)
        let wave44k = resample(samples, from: sampleRate, to: Self.sampleRate)
        let align = VAEEncoderWeights.frame * VAEEncoderWeights.patch
        let remainder = wave44k.count % align
        let padded = remainder == 0 ? wave44k : [Float](repeating: 0, count: align - remainder) + wave44k
        let (mean, std) = try engine.encodeLatents(samples: padded)
        let draw = noise ?? MingEngine.gaussianNoise(count: mean.count, seed: seed)
        guard draw.count == mean.count else { throw MingTTSError.badInput("posterior noise has \(draw.count) values, the clip has \(mean.count) latent values") }
        let latents = (0..<mean.count).map { mean[$0] + std[$0] * draw[$0] }
        return try MingReferenceVoice(transcript: transcript, latents: latents, speakerEmbedding: embedding)
    }

    private func speakerNet() throws -> CAMPlus {
        speakerNetLock.lock()
        defer { speakerNetLock.unlock() }
        if let speakerNetCache { return speakerNetCache }
        let net = try CAMPlus(onnx: directory.appendingPathComponent("campplus.onnx"))
        speakerNetCache = net
        return net
    }

    /// The file's samples mixed to mono at its own rate.
    static func readMono(_ url: URL) throws -> (samples: [Float], sampleRate: Int) {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw MingTTSError.badInput("\(url.lastPathComponent) holds no audio")
        }
        try file.read(into: buffer)
        guard let channels = buffer.floatChannelData else { throw MingTTSError.badInput("\(url.lastPathComponent) is not decodable to float PCM") }
        let count = Int(buffer.frameLength), channelCount = Int(format.channelCount)
        var mono = [Float](repeating: 0, count: count)
        for c in 0..<channelCount {
            let data = channels[c]
            for i in 0..<count { mono[i] += data[i] }
        }
        if channelCount > 1 { for i in 0..<count { mono[i] /= Float(channelCount) } }
        return (mono, Int(format.sampleRate))
    }

    // MARK: parity

    /// Teacher-forced parity run from golden prompt embeddings and noise. `teacherLatents`, when
    /// given, replace the sampled latents downstream of the flow head so each stage is checked
    /// against the reference inputs rather than a drifted autoregressive state. `initialHistory` is
    /// the step 0 latent history of a clone prompt.
    public func parityRun(promptEmbeddings: [Float], noise: [Float], steps: Int, cfg: Float, teacherLatents: [[Float]]?, initialHistory: [Float]? = nil) throws -> (capture: ParityCapture, pcm: [Float], report: GenerationReport) {
        let capture = ParityCapture()
        capture.teacherLatents = teacherLatents
        var o = MingEngine.Options()
        o.cfg = cfg
        o.maxSteps = steps
        o.noise = noise
        o.initialHistory = initialHistory
        var pcm: [Float] = []
        let outcome = try engine.generate(embeds: promptEmbeddings, options: o, capture: capture) { pcm.append(contentsOf: $0) }
        return (capture, pcm, GenerationReport(outcome: outcome))
    }

    /// Prompt embeddings as the engine feeds them: token rows, the speaker row, the reference rows.
    public func promptEmbeddings(_ prompt: MingPrompt) throws -> [Float] {
        try engine.promptEmbeddings(prompt: prompt)
    }

    /// Posterior mean and std of a 44.1 kHz waveform that is a whole number of 3528-sample patches.
    func encodeLatents(samples: [Float]) throws -> (mean: [Float], std: [Float]) {
        try engine.encodeLatents(samples: samples)
    }
}

/// What one utterance took.
public struct GenerationReport: Sendable {
    public var steps: Int
    public var stopped: Bool
    public var prefillSeconds: Double
    /// Wall time per step, from encode start to GPU completion.
    public var stepWallSeconds: [Double]
    /// GPU time per stage per step; one "step" stage per step unless the run was stage-split.
    public var stepTimings: [StepTimings]

    init(outcome: MingEngine.Outcome) {
        steps = outcome.steps
        stopped = outcome.stopped
        prefillSeconds = outcome.prefillSeconds
        stepWallSeconds = outcome.wallSeconds
        stepTimings = outcome.stepTimings
    }
}
