// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import MingTTS

// ming-tts: parity against a golden case, speech to a wav, or a benchmark.
//   ming-tts --parity <golden case dir> [--weights dir]
//   ming-tts --say "text" [task] [--language en|none|ja|...] [--speaker zero|none] [--clone ref.wav --transcript "..." | --anchor] [--out file.wav] [--weights dir]
//     task: --voice US_Emma (the default) | --plain | --style "voice description" | --instruction "free-form direction"
//           | --emotion happy|sad|angry|fearful|disgusted|surprised|neutral | --speed/--pitch/--volume low|medium|high
//           | --dialect cantonese|chuanyu
//   ming-tts --bench [--text "..."] [--repeat n] [--clone ref.wav --transcript "..."] [--weights dir]

struct Args {
    var parity: String?
    var say: String?
    var voice: String?
    var plain = false
    var style: String?
    var instruction: String?
    var emotion: String?
    var speed: String?
    var pitch: String?
    var volume: String?
    var dialect: String?
    var language: String?
    var speaker = "zero"
    var clone: String?
    var anchor = false
    var transcript: String?
    var out = "out.wav"
    var bench = false
    var gemvBench = false
    var qwenBench = false
    var simdLayout = false
    var benchText = "The garden will remain open until sunset, provided the weather stays clear, and the volunteers who tend the roses will stay to answer questions from visitors."
    var runs = 3
    var weights = ProcessInfo.processInfo.environment["MING_WEIGHTS"]
    var seed: UInt64 = 0

    init(_ argv: [String]) throws {
        var i = 1
        func value() throws -> String {
            i += 1
            guard i < argv.count else { throw MingTTSError.badInput("\(argv[i - 1]) needs a value") }
            return argv[i]
        }
        while i < argv.count {
            switch argv[i] {
            case "--parity": parity = try value()
            case "--say": say = try value()
            case "--voice": voice = try value()
            case "--plain": plain = true
            case "--style": style = try value()
            case "--instruction": instruction = try value()
            case "--emotion": emotion = try value()
            case "--speed": speed = try value()
            case "--pitch": pitch = try value()
            case "--volume": volume = try value()
            case "--dialect": dialect = try value()
            case "--language": language = try value()
            case "--speaker": speaker = try value()
            case "--clone": clone = try value()
            case "--anchor": anchor = true
            case "--transcript": transcript = try value()
            case "--out": out = try value()
            case "--bench": bench = true
            case "--gemv-bench": gemvBench = true
            case "--qwen-bench": qwenBench = true
            case "--simd-layout": simdLayout = true
            case "--text": benchText = try value()
            case "--repeat": runs = Int(try value()) ?? 3
            case "--weights": weights = try value()
            case "--seed": seed = UInt64(try value()) ?? 0
            default: throw MingTTSError.badInput("unknown argument \(argv[i])")
            }
            i += 1
        }
    }
}

extension Args {
    /// The request the task flags describe. Each flag implies its task; flags of two tasks conflict.
    func request(text: String, model: MingTTSModel) throws -> MingRequest {
        func parse<T: CaseIterable>(_ flag: String, _ value: String?) throws -> T? {
            guard let value else { return nil }
            guard let match = T.allCases.first(where: { "\($0)" == value }) else {
                throw MingTTSError.badInput("--\(flag) takes \(T.allCases.map { "\($0)" }.joined(separator: "|")), not \(value)")
            }
            return match
        }
        let speed: MingLevel? = try parse("speed", speed), pitch: MingLevel? = try parse("pitch", pitch), volume: MingLevel? = try parse("volume", volume)
        let tasks: [MingTask] = [
            plain ? .plain : nil,
            voice.map { .builtInVoice(MingVoice(ip: $0)) },
            style.map { .voiceDesign($0) },
            instruction.map { .freeform($0) },
            try (parse("emotion", emotion) as MingEmotion?).map { .emotion($0) },
            [speed, pitch, volume].contains { $0 != nil } ? .attributes(speed: speed, pitch: pitch, volume: volume) : nil,
            try (parse("dialect", dialect) as MingDialect?).map { .dialect($0) },
        ].compactMap { $0 }
        guard tasks.count <= 1 else { throw MingTTSError.badInput("give the flags of one task only") }
        let task = tasks.first ?? .builtInVoice(.emma)
        let isDialect = if case .dialect = task { true } else { false }
        let language: MingLanguage?
        switch self.language {
        case "none": language = nil
        case nil: language = isDialect ? nil : .english
        case let id?:
            guard let parsed = MingLanguage(id: id) else { throw MingTTSError.badInput("unknown language \(id)") }
            language = parsed
        }
        let conditioning: MingConditioning
        if anchor {
            guard case .builtInVoice(let voice) = task, clone == nil else { throw MingTTSError.badInput("--anchor needs --voice and no --clone") }
            return try MingRequest(text: text, task: task, language: language, conditioning: .clone(try model.anchor(voice)))
        }
        switch (self.speaker, clone) {
        case (_, let wav?):
            guard let transcript else { throw MingTTSError.badInput("--clone needs --transcript") }
            conditioning = .clone(try model.encodeReference(url: URL(fileURLWithPath: wav), transcript: transcript, seed: seed),
                                  speakerEmbedding: speaker != "none")
        case ("zero", nil): conditioning = .zeroSpeaker
        case ("none", nil): conditioning = .noSpeaker
        default: throw MingTTSError.badInput("--speaker takes zero|none")
        }
        return try MingRequest(text: text, task: task, language: language, conditioning: conditioning)
    }
}

func writeWav(_ samples: [Float], to path: String, sampleRate: Int = MingTTSModel.sampleRate) throws {
    var d = Data()
    func u32(_ v: UInt32) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 4)) }
    func u16(_ v: UInt16) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 2)) }
    let bytes = samples.count * 4
    d.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + bytes)); d.append("WAVE".data(using: .ascii)!)
    d.append("fmt ".data(using: .ascii)!); u32(16); u16(3); u16(1); u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 4)); u16(4); u16(32)
    d.append("data".data(using: .ascii)!); u32(UInt32(bytes))
    samples.withUnsafeBytes { d.append(contentsOf: $0) }
    try d.write(to: URL(fileURLWithPath: path))
}

func readF32(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

/// Max abs error and cosine similarity of two equally long vectors.
func compare(_ got: [Float], _ want: [Float]) -> (maxAbs: Float, cosine: Double, count: Int) {
    let n = min(got.count, want.count)
    var maxAbs: Float = 0, dot = 0.0, na = 0.0, nb = 0.0
    for i in 0..<n {
        maxAbs = max(maxAbs, abs(got[i] - want[i]))
        dot += Double(got[i]) * Double(want[i])
        na += Double(got[i]) * Double(got[i])
        nb += Double(want[i]) * Double(want[i])
    }
    return (maxAbs, dot / max((na * nb).squareRoot(), 1e-30), n)
}

func fmt(_ x: Double, _ digits: Int = 3) -> String { String(format: "%.\(digits)f", x) }

func loadModel(_ args: Args) throws -> MingTTSModel {
    guard let weights = args.weights else { throw MingTTSError.badInput("pass --weights dir or set MING_WEIGHTS") }
    let t0 = Date()
    let model = try MingTTSModel(directory: URL(fileURLWithPath: weights))
    print("loaded \(weights) in \(fmt(Date().timeIntervalSince(t0), 2)) s")
    return model
}

func runParity(_ args: Args, dir: String) throws -> Bool {
    let caseURL = URL(fileURLWithPath: dir)
    let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: caseURL.appendingPathComponent("manifest.json"))) as! [String: Any]
    let tensors = manifest["tensors"] as! [String: [String: Any]]
    func golden(_ name: String) throws -> [Float] { try readF32(caseURL.appendingPathComponent(tensors[name]!["file"] as! String)) }
    let steps = manifest["steps"] as! Int
    let cfg = Float(manifest["cfg"] as! Double)
    let embeds = try golden("inputs_embeds")
    let noise = try golden("noise")
    let goldLatents = try golden("latents")
    let latentRows = stride(from: 0, to: goldLatents.count, by: 64).map { Array(goldLatents[$0..<($0 + 64)]) }
    let model = try loadModel(args)

    print("case \(dir): \(steps) steps, prompt \(embeds.count / 896) tokens")
    var allPass = true
    // The engine's own prompt path against the golden ids and embeddings.
    let goldIDs = (manifest["input_ids"] as! [Int]).map(Int32.init)
    // A clone case (manifest prompt_text) is task TTS with the golden reference latents and speaker embedding.
    let language = (manifest["language_id"] as? String).flatMap(MingLanguage.init(id:))
    let goldRequest: MingRequest
    var initialHistory: [Float]?
    if let transcript = manifest["prompt_text"] as? String {
        let voice = try MingReferenceVoice(transcript: transcript, latents: try golden("prompt_latent"), speakerEmbedding: try golden("spk_emb"))
        goldRequest = try MingRequest(text: manifest["text"] as! String, task: .plain, language: language, conditioning: .clone(voice))
        initialHistory = Array(voice.latents.suffix(64))
    } else {
        goldRequest = try MingRequest(text: manifest["text"] as! String, task: .builtInVoice(MingVoice(ip: manifest["voice"] as! String)),
                                      language: language, conditioning: .zeroSpeaker)
    }
    let ownPrompt = model.prompt(fragment: goldRequest.text, of: goldRequest)
    let idsMatch = ownPrompt.ids == goldIDs && ownPrompt.speakerPositions == (manifest["speaker_positions"] as! [Int])
        && (ownPrompt.latentPosition ?? -1) == (manifest["latent_position"] as? Int ?? -1)
    allPass = allPass && idsMatch
    print("  \(idsMatch ? "ok  " : "FAIL") prompt ids             \(ownPrompt.ids.count) tokens vs golden \(goldIDs.count), speaker rows \(ownPrompt.speakerPositions), latent row \(ownPrompt.latentPosition.map(String.init) ?? "none")")
    if !idsMatch, let first = zip(ownPrompt.ids, goldIDs).enumerated().first(where: { $0.element.0 != $0.element.1 }) {
        print("       first difference at \(first.offset): own \(first.element.0) golden \(first.element.1)")
    }
    let ownEmbeds = try model.promptEmbeddings(ownPrompt)
    func report(_ name: String, _ got: [Float], _ want: [Float], cosineFloor: Double = 0.9999) {
        let c = compare(got, want)
        let sizeNote = got.count == want.count ? "" : "  (count \(got.count) vs \(want.count))"
        let pass = c.cosine >= cosineFloor && got.count == want.count
        allPass = allPass && pass
        print("  \(pass ? "ok  " : "FAIL") \(name.padding(toLength: 22, withPad: " ", startingAt: 0)) maxabs \(String(format: "%.3e", c.maxAbs))  cosine \(fmt(c.cosine, 6))\(sizeNote)")
    }

    report("inputs_embeds", ownEmbeds, embeds)
    print("teacher-forced:")
    let t0 = Date()
    let tf = try model.parityRun(promptEmbeddings: embeds, noise: noise, steps: steps, cfg: cfg, teacherLatents: latentRows, initialHistory: initialHistory)
    print("  ran \(tf.report.steps) steps in \(fmt(Date().timeIntervalSince(t0), 2)) s, stopped \(tf.report.stopped)")
    let cap = tf.capture
    report("llm_layer0_prefill", cap.llmLayer0Prefill, try golden("llm_layer0_prefill"))
    report("llm_layer23_prefill", cap.llmLayer23Prefill, try golden("llm_layer23_prefill"))
    report("llm_hidden", Array(cap.llmHidden.joined()), try golden("llm_hidden"))
    report("stop_prob", cap.stopProb, try golden("stop_prob"), cosineFloor: 0.999)
    report("dit_velocity_step0", cap.velocityStep0, try golden("dit_velocity_step0"))
    report("latents", Array(cap.latents.joined()), goldLatents)
    report("agg_out", Array(cap.aggOut.joined()), try golden("agg_out"))
    report("vae_hidden", Array(cap.vaeHidden.joined()), try golden("vae_hidden"))
    report("waveform_stream", tf.pcm, try golden("waveform_stream"))
    report("waveform", tf.pcm, try golden("waveform"))
    let goldStop = try golden("stop_prob")
    let firstStop = cap.stopProb.enumerated().first { $0.offset > 3 && $0.element > 0.5 }?.offset
    print("  stop at step \(firstStop.map(String.init) ?? "none") (golden \(steps - 1)); max |stop_prob diff| \(String(format: "%.3e", zip(cap.stopProb, goldStop).map { abs($0 - $1) }.max() ?? 0))")

    print("end to end with golden noise (autoregressive):")
    let e2e = try model.parityRun(promptEmbeddings: embeds, noise: noise, steps: steps, cfg: cfg, teacherLatents: nil, initialHistory: initialHistory)
    let ref = try golden("waveform")
    print("  \(e2e.report.steps) steps (golden \(steps)), stopped \(e2e.report.stopped); \(fmt(Double(e2e.pcm.count) / 44100, 2)) s audio vs \(fmt(Double(ref.count) / 44100, 2)) s")
    let c = compare(e2e.pcm, ref)
    print("  waveform maxabs \(String(format: "%.3e", c.maxAbs)) cosine \(fmt(c.cosine, 4)) over \(c.count) samples (drift expected after many steps)")
    print("  latent cosine per step:", e2e.capture.latents.prefix(min(steps, 12)).enumerated().map { fmt(compare($0.element, latentRows[$0.offset]).cosine, 4) }.joined(separator: " "))
    try writeWav(e2e.pcm, to: args.out)
    print("  wrote \(args.out)")
    print(allPass ? "PARITY PASS" : "PARITY FAIL")
    return allPass
}

func runSay(_ args: Args, text: String) throws {
    let model = try loadModel(args)
    let tEncode = Date()
    let request = try args.request(text: text, model: model)
    if args.clone != nil { print("encoded the reference clip in \(fmt(Date().timeIntervalSince(tEncode) * 1000, 0)) ms") }
    var pcm: [Float] = []
    var fragmentPCM: [[Float]] = []
    let t0 = Date()
    var firstAudio: Double?
    var options = SynthesisOptions()
    options.seed = args.seed
    let reports = try model.synthesize(request, options: options) { fragment, chunk in
        if firstAudio == nil, !chunk.isEmpty { firstAudio = Date().timeIntervalSince(t0) }
        pcm.append(contentsOf: chunk)
        while fragmentPCM.count <= fragment { fragmentPCM.append([]) }
        fragmentPCM[fragment].append(contentsOf: chunk)
    }
    for report in reports {
        print("fragment: \(report.steps) steps, stopped \(report.stopped), prefill \(fmt(report.prefillSeconds * 1000, 1)) ms")
    }
    let wall = Date().timeIntervalSince(t0)
    let audio = Double(pcm.count) / Double(MingTTSModel.sampleRate)
    print("\(fmt(audio, 2)) s of audio in \(fmt(wall, 2)) s wall (RTF \(fmt(wall / audio, 3))), first audio after \(fmt((firstAudio ?? wall) * 1000, 0)) ms, peak \(fmt(Double(pcm.map(abs).max() ?? 0), 3))")
    try writeWav(pcm, to: args.out)
    print("wrote \(args.out)")
    if fragmentPCM.count > 1 {
        for (i, samples) in fragmentPCM.enumerated() {
            let path = args.out.replacingOccurrences(of: ".wav", with: ".frag\(i + 1).wav")
            try writeWav(samples, to: path)
            print("wrote \(path)")
        }
    }
}

func runBench(_ args: Args) throws {
    let model = try loadModel(args)
    let request = try args.request(text: args.benchText, model: model)
    let p = model.prompt(fragment: request.text, of: request)
    print("bench: \(p.ids.count) prompt tokens, \(args.runs) runs, GPU time per stage from command buffer GPUStartTime/GPUEndTime")
    for run in 0..<args.runs {
        var pcm: [Float] = []
        let t0 = Date()
        var firstAudio: Double?
        var options = SynthesisOptions()
        options.seed = args.seed
        let report = try model.synthesize(prompt: p, options: options, splitStages: true) { chunk in
            if firstAudio == nil, !chunk.isEmpty { firstAudio = Date().timeIntervalSince(t0) }
            pcm.append(contentsOf: chunk)
        }
        let wall = Date().timeIntervalSince(t0)
        let audio = Double(pcm.count) / Double(MingTTSModel.sampleRate)
        let stepWall = report.stepWallSeconds.reduce(0, +)
        var stage: [String: Double] = [:]
        for t in report.stepTimings { for (name, s) in t.stages { stage[name, default: 0] += s } }
        let gpuTotal = stage.values.reduce(0, +)
        let order = ["llm", "flow", "agg", "vae"]
        let split = order.map { "\($0) \(fmt(stage[$0, default: 0] / Double(report.steps) * 1000, 2))" }.joined(separator: "  ")
        print("run \(run): \(report.steps) steps, \(fmt(audio, 2)) s audio, wall \(fmt(wall, 3)) s, RTF \(fmt(wall / audio, 3)), \(fmt(Double(report.steps) / stepWall, 1)) steps/s, first audio \(fmt((firstAudio ?? wall) * 1000, 0)) ms, prefill \(fmt(report.prefillSeconds * 1000, 1)) ms")
        print("        per step ms: wall \(fmt(stepWall / Double(report.steps) * 1000, 2))  gpu \(fmt(gpuTotal / Double(report.steps) * 1000, 2))  [\(split)]")
    }
}

do {
    let args = try Args(CommandLine.arguments)
    if let dir = args.parity {
        exit(try runParity(args, dir: dir) ? 0 : 1)
    } else if let text = args.say {
        try runSay(args, text: text)
    } else if args.bench {
        try runBench(args)
    } else if args.simdLayout {
        try GemvBench.layout()
    } else if args.gemvBench {
        try GemvBench.run(repeats: args.runs)
    } else if args.qwenBench {
        try QwenBench.run(rounds: args.runs)
    } else {
        print("usage: ming-tts --parity <case dir> | --say \"text\" [--voice V | --plain | --style S | --instruction I | --emotion E | --speed/--pitch/--volume L | --dialect D] [--language L|none] [--speaker zero|none] [--out f.wav] | --bench [--text t] [--repeat n]   (--weights dir, or MING_WEIGHTS)")
    }
} catch {
    FileHandle.standardError.write("ming-tts: \(error)\n".data(using: .utf8)!)
    exit(2)
}
