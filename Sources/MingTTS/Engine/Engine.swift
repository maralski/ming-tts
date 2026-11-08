// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Everything a generation records for parity checks against the reference goldens.
public final class ParityCapture {
    public var llmLayer0Prefill: [Float] = []
    public var llmLayer23Prefill: [Float] = []
    public var llmHidden: [[Float]] = []
    public var stopProb: [Float] = []
    /// [flowStep][cond, uncond][64] DiT outputs of generation step 0.
    public var velocityStep0: [Float] = []
    public var latents: [[Float]] = []
    public var aggOut: [[Float]] = []
    /// Decoder final-norm rows, four per decoded latent, in order.
    public var vaeHidden: [[Float]] = []
    /// Golden latents to feed the aggregator, the VAE and the next step's history instead of
    /// the sampled ones (teacher forcing). Sampled latents are still recorded in `latents`.
    public var teacherLatents: [[Float]]?
    public init() {}
}

/// GPU time per stage of one generation step, seconds.
public struct StepTimings: Sendable {
    public var stages: [(String, Double)] = []
    public var total: Double { stages.reduce(0) { $0 + $1.1 } }
}

/// One generation step's command buffers: one per stage when `split` (for timing), else one.
final class StepRecorder {
    let ctx: MetalContext
    let kernels: Kernels
    let split: Bool
    private var current: (name: String, cb: MTLCommandBuffer, ce: MTLComputeCommandEncoder, concurrent: Bool)?
    private var done: [(String, MTLCommandBuffer)] = []

    init(ctx: MetalContext, kernels: Kernels, split: Bool) {
        self.ctx = ctx
        self.kernels = kernels
        self.split = split
    }

    /// A concurrent stage runs its dispatches at once unless the caller places barriers
    /// (`Encoder.encodePaired` does); a serial stage orders them. When stages share one command
    /// buffer, a change of kind starts a new encoder on it.
    func stage(_ name: String, concurrent: Bool = false) -> Encoder {
        var cb: MTLCommandBuffer
        var label = name
        if let cur = current {
            if !split && cur.concurrent == concurrent { return Encoder(k: kernels, ce: cur.ce) }
            cur.ce.endEncoding()
            if split {
                cur.cb.commit()
                done.append((cur.name, cur.cb))
                cb = ctx.queue.makeCommandBuffer()!
            } else {
                cb = cur.cb
                label = cur.name
            }
        } else {
            cb = ctx.queue.makeCommandBuffer()!
        }
        let ce = cb.makeComputeCommandEncoder(dispatchType: concurrent ? .concurrent : .serial)!
        current = (label, cb, ce, concurrent)
        return Encoder(k: kernels, ce: ce)
    }

    /// Commits everything and waits. Returns per-stage GPU times.
    func finish() throws -> StepTimings {
        if let cur = current {
            cur.ce.endEncoding()
            cur.cb.commit()
            done.append((cur.name, cur.cb))
            current = nil
        }
        var t = StepTimings()
        for (name, cb) in done {
            cb.waitUntilCompleted()
            if let e = cb.error { throw MingTTSError.gpuFailure("\(name): \(e)") }
            t.stages.append((name, cb.gpuEndTime - cb.gpuStartTime))
        }
        done.removeAll()
        return t
    }
}

/// The Ming-3.0-tts generation loop on Metal: prompt prefill, then per step the LLM token, the
/// stop head, the CFM flow head, the aggregator and the streaming VAE decoder.
final class MingEngine {
    static let llmCapacity = 512
    static let latentDim = 64

    let ctx: MetalContext
    let kernels: Kernels
    let st: SafeTensors
    let llm: LLMWeights
    let flowWeights: FlowHeadWeights
    let aggWeights: AggregatorWeights
    let vaeWeights: VAEDecoderWeights

    let llmCache: KVCache
    let llmForward: QwenForward
    let xLLM: MTLBuffer
    let z: MTLBuffer
    let stopLogits: MTLBuffer
    let layerCapture: [MTLBuffer]

    let flow: DiTForward
    let agg: DiTForward
    let temb: MTLBuffer
    let cemb: MTLBuffer
    let y: MTLBuffer
    let history: MTLBuffer
    let ditX: MTLBuffer
    let ditOut: MTLBuffer
    let velocity: MTLBuffer
    let aggX: MTLBuffer
    let cls: MTLBuffer
    let noise: MTLBuffer
    let latents: MTLBuffer
    let teacher: MTLBuffer

    let vae: VAEDecoder
    /// `spk_head` as host floats: [896, 192] and its bias. One 896-vector per prompt is CPU work.
    let spkHeadWeight: [Float]
    let spkHeadBias: [Float]
    let timesteps: [Float]
    let flowSteps = 10
    let maxSteps = 512

    /// What only a clone prompt needs, built on first use: the VAE encoder and an aggregator pass
    /// wide enough for every reference latent at once.
    private var cloneSupportCache: CloneSupport?
    private struct CloneSupport {
        let encoder: VAEEncoder
        let aggregator: DiTForward
        let aggX: MTLBuffer
        let aggOut: MTLBuffer
        let latents: MTLBuffer
    }

    init(directory: URL) throws {
        ctx = try MetalContext()
        kernels = try Kernels(ctx: ctx)
        let st = try SafeTensors(url: directory.appendingPathComponent("model.safetensors"))
        self.st = st
        let ctx = self.ctx
        llm = try WeightArena.load(st, ctx: ctx, label: "llm") { try LLMWeights($0) }
        flowWeights = try WeightArena.load(st, ctx: ctx, label: "flow") { try FlowHeadWeights($0, st: st) }
        aggWeights = try WeightArena.load(st, ctx: ctx, label: "agg") { try AggregatorWeights($0, st: st) }
        vaeWeights = try WeightArena.load(st, ctx: ctx, label: "vae") { try VAEDecoderWeights($0, st: st) }

        let cfg = llm.stack.config
        llmCache = try KVCache(ctx: ctx, config: cfg, capacity: Self.llmCapacity, label: "llm.cache")
        llmForward = try QwenForward(ctx: ctx, weights: llm.stack, cache: llmCache,
                                     work: QwenWorkspace(ctx: ctx, config: cfg, maxRows: Self.llmCapacity, label: "llm"))
        xLLM = try ctx.makeBuffer(bytes: (Self.llmCapacity + 8) * cfg.hidden * 4, label: "llm.x")
        z = try ctx.makeBuffer(bytes: 8 * cfg.hidden * 4, label: "llm.z")
        stopLogits = try ctx.makeBuffer(bytes: 16 * 4, label: "llm.stop")
        layerCapture = try (0..<2).map { try ctx.makeBuffer(bytes: Self.llmCapacity * cfg.hidden * 4, label: "llm.capture\($0)") }

        let h = DiTWeights.hidden
        flow = try DiTForward(ctx: ctx, weights: flowWeights.dit, work: DiTWorkspace(ctx: ctx, maxRows: 6, label: "flow"))
        agg = try DiTForward(ctx: ctx, weights: aggWeights.dit, work: DiTWorkspace(ctx: ctx, maxRows: 2, label: "agg"))
        temb = try ctx.makeBuffer(bytes: (flowSteps + 8) * h * 4, label: "flow.temb")
        cemb = try ctx.makeBuffer(bytes: 8 * h * 4, label: "flow.cemb")
        y = try ctx.makeBuffer(bytes: 8 * Self.latentDim * 4, label: "flow.y")
        history = try ctx.makeBuffer(bytes: 8 * Self.latentDim * 4, label: "flow.history")
        ditX = try ctx.makeBuffer(bytes: 14 * h * 4, label: "flow.x")
        ditOut = try ctx.makeBuffer(bytes: 14 * cfg.hidden * 4, label: "dit.out")
        velocity = try ctx.makeBuffer(bytes: flowSteps * 2 * Self.latentDim * 4, label: "flow.velocity")
        aggX = try ctx.makeBuffer(bytes: 10 * h * 4, label: "agg.x")
        cls = try ctx.makeBuffer(floats: aggWeights.cls, label: "agg.cls")
        noise = try ctx.makeBuffer(bytes: maxSteps * Self.latentDim * 4, label: "flow.noise")
        latents = try ctx.makeBuffer(bytes: maxSteps * Self.latentDim * 4, label: "latents")
        teacher = try ctx.makeBuffer(bytes: maxSteps * Self.latentDim * 4, label: "teacher")

        vae = try VAEDecoder(ctx: ctx, weights: vaeWeights, maxLatents: maxSteps)
        spkHeadWeight = try st.floats("spk_head.weight")
        spkHeadBias = try st.floats("spk_head.bias")
        timesteps = FlowSchedule.timesteps(steps: flowSteps)
        try precomputeTimeEmbeddings()
    }

    /// t_embedder of every flow timestep is input-independent, so it is computed once.
    private func precomputeTimeEmbeddings() throws {
        let sinus = try ctx.makeBuffer(bytes: (flowSteps + 8) * 256 * 4, label: "flow.sinus")
        for (k, t) in timesteps.prefix(flowSteps).enumerated() {
            sinus.write(FlowSchedule.sinusoid(t), offset: k * 256 * 4)
        }
        let mid = try ctx.makeBuffer(bytes: (flowSteps + 8) * 1024 * 4, label: "flow.tmid")
        let rec = StepRecorder(ctx: ctx, kernels: kernels, split: false)
        let e = rec.stage("temb")
        e.linear(ActView(sinus, ld: 256), rows: flowSteps, flowWeights.time1, ActView(mid, ld: 1024), flags: .silu)
        e.linear(ActView(mid, ld: 1024), rows: flowSteps, flowWeights.time2, ActView(temb, ld: 1024))
        _ = try rec.finish()
    }

    // MARK: prompt

    /// The prompt as the LLM reads it: token rows from the embedding table, the speaker row zeros or
    /// `spk_head` of the speaker embedding, and for a clone one aggregator row per reference latent.
    func promptEmbeddings(prompt: MingPrompt) throws -> [Float] {
        let hidden = llm.stack.config.hidden
        var out = [Float](repeating: 0, count: prompt.ids.count * hidden)
        let latentRows = prompt.reference.map { prompt.latentPosition!..<(prompt.latentPosition! + $0.latentRows) } ?? 0..<0
        let replaced = Set(prompt.speakerPositions).union(latentRows)
        for (i, id) in prompt.ids.enumerated() where !replaced.contains(i) {
            let row = try st.bf16Row("model.model.embed_tokens.weight", index: Int(id))
            out.replaceSubrange((i * hidden)..<((i + 1) * hidden), with: row)
        }
        if case .clone(let voice, true) = prompt.conditioning {
            let row = speakerRow(voice.speakerEmbedding)
            for position in prompt.speakerPositions { out.replaceSubrange((position * hidden)..<((position + 1) * hidden), with: row) }
        }
        if let voice = prompt.reference {
            let rows = try aggregate(latents: voice.latents)
            out.replaceSubrange((latentRows.lowerBound * hidden)..<(latentRows.upperBound * hidden), with: rows)
        }
        return out
    }

    /// `spk_head(embedding)`: 192 -> 896.
    func speakerRow(_ embedding: [Float]) -> [Float] {
        let k = MingReferenceVoice.speakerEmbeddingDim
        return (0..<spkHeadBias.count).map { n in
            var acc = spkHeadBias[n]
            for i in 0..<k { acc += spkHeadWeight[n * k + i] * embedding[i] }
            return acc
        }
    }

    /// `linear_proj_audio` over every reference latent at once: the aggregator with batch P, seq 2
    /// ([cls, x_embedder(latent)] per latent), output row 0 of each sequence.
    func aggregate(latents: [Float]) throws -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        let support = try cloneSupport()
        let h = DiTWeights.hidden, hidden = llm.stack.config.hidden, dim = Self.latentDim
        let count = latents.count / dim
        guard count <= Self.llmCapacity else { throw MingTTSError.badInput("\(count) reference latents exceed the \(Self.llmCapacity) position cache") }
        var x = [Float](repeating: 0, count: 2 * count * h)
        for p in 0..<count { x.replaceSubrange((2 * p * h)..<((2 * p + 1) * h), with: aggWeights.cls) }
        support.aggX.write(x)
        support.latents.write(latents)
        let rec = StepRecorder(ctx: ctx, kernels: kernels, split: false)
        let e = rec.stage("prompt.agg")
        e.linear(ActView(support.latents, ld: dim), rows: count, aggWeights.xEmbedder, ActView(support.aggX, offset: h * 4, ld: 2 * h))
        support.aggregator.encode(e, x: ActView(support.aggX, ld: h), batch: count, seq: 2, outRows: 2 * count, out: ActView(support.aggOut, ld: hidden))
        _ = try rec.finish()
        let all = support.aggOut.floats(count: 2 * count * hidden)
        var rows = [Float](repeating: 0, count: count * hidden)
        for p in 0..<count { rows.replaceSubrange((p * hidden)..<((p + 1) * hidden), with: all[(2 * p * hidden)..<((2 * p + 1) * hidden)]) }
        return rows
    }

    /// Posterior mean and std of a padded 44.1 kHz waveform, one row of 64 each per 3528 samples.
    func encodeLatents(samples: [Float]) throws -> (mean: [Float], std: [Float]) {
        lock.lock()
        defer { lock.unlock() }
        return try cloneSupport().encoder.encode(samples: samples)
    }

    private func cloneSupport() throws -> CloneSupport {
        if let cloneSupportCache { return cloneSupportCache }
        let encoderWeights = try WeightArena.load(st, ctx: ctx, label: "vae.encoder") { try VAEEncoderWeights($0, st: st) }
        let rows = 2 * Self.llmCapacity
        let support = CloneSupport(
            encoder: try VAEEncoder(ctx: ctx, kernels: kernels, weights: encoderWeights),
            aggregator: try DiTForward(ctx: ctx, weights: aggWeights.dit, work: DiTWorkspace(ctx: ctx, maxRows: rows, label: "prompt.agg")),
            aggX: try ctx.makeBuffer(bytes: (rows + 8) * DiTWeights.hidden * 4, label: "prompt.agg.x"),
            aggOut: try ctx.makeBuffer(bytes: (rows + 8) * llm.stack.config.hidden * 4, label: "prompt.agg.out"),
            latents: try ctx.makeBuffer(bytes: (Self.llmCapacity + 8) * Self.latentDim * 4, label: "prompt.agg.latents"))
        cloneSupportCache = support
        return support
    }

    // MARK: generation

    struct Options {
        var cfg: Float = 2.0
        var maxSteps = 1000
        var seed: UInt64 = 0
        var noise: [Float]?
        /// The latent history at step 0: the last reference latent of a clone prompt, else zeros.
        var initialHistory: [Float]?
        var splitStages = false
    }

    struct Outcome {
        var steps = 0
        var stopped = false
        var prefillSeconds = 0.0
        var stepTimings: [StepTimings] = []
        var wallSeconds: [Double] = []
    }

    private let lock = NSLock()

    /// Runs one utterance. `onChunk` receives PCM as each chunk completes on the GPU.
    func generate(embeds: [Float], options: Options, capture: ParityCapture? = nil, onChunk: ([Float]) throws -> Void) throws -> Outcome {
        lock.lock()
        defer { lock.unlock() }
        let hidden = llm.stack.config.hidden
        let promptRows = embeds.count / hidden
        guard promptRows >= 1, promptRows * hidden == embeds.count else { throw MingTTSError.badInput("prompt embeddings are not [T, \(hidden)]") }
        let budget = (Self.llmCapacity - promptRows)
        guard budget >= 1 else { throw MingTTSError.badInput("prompt of \(promptRows) tokens leaves no room in the \(Self.llmCapacity) position cache") }
        let steps = min(options.maxSteps, budget, maxSteps)

        llmCache.reset()
        vae.reset()
        memset(history.contents(), 0, history.length)
        if let initialHistory = options.initialHistory { history.write(initialHistory) }
        memset(stopLogits.contents(), 0, stopLogits.length)
        noise.write(options.noise ?? Self.gaussianNoise(count: steps * Self.latentDim, seed: options.seed))
        if let t = capture?.teacherLatents {
            teacher.write(Array(t.joined()))
        }
        var outcome = Outcome()

        // Prefill.
        xLLM.write(embeds)
        let start = Date()
        let rec = StepRecorder(ctx: ctx, kernels: kernels, split: false)
        let e = rec.stage("prefill")
        let x = ActView(xLLM, ld: hidden)
        llmForward.encode(e, x: x, rows: promptRows, normRows: 1, out: ActView(z, ld: hidden)) { layer, x in
            guard capture != nil, layer == 0 || layer == 23 else { return }
            e.copy(x, count: promptRows * hidden, to: ActView(self.layerCapture[layer == 0 ? 0 : 1], ld: hidden))
        }
        _ = try rec.finish()
        outcome.prefillSeconds = Date().timeIntervalSince(start)
        if let capture {
            capture.llmLayer0Prefill = layerCapture[0].floats(count: promptRows * hidden)
            capture.llmLayer23Prefill = layerCapture[1].floats(count: promptRows * hidden)
        }

        for n in 0..<steps {
            let stepStart = Date()
            let rec = StepRecorder(ctx: ctx, kernels: kernels, split: options.splitStages)
            let chunk = encodeStep(n, rec: rec, cfg: options.cfg, useTeacher: capture?.teacherLatents != nil)
            let timings = try rec.finish()
            outcome.wallSeconds.append(Date().timeIntervalSince(stepStart))
            outcome.stepTimings.append(timings)
            outcome.steps = n + 1
            let logits = stopLogits.floats(count: 2, offset: (n % 2) * 8 * 4)
            let stopProb = 1 / (1 + exp(logits[0] - logits[1]))
            if let capture { record(capture, step: n, stopProb: stopProb, chunk: chunk) }
            if chunk.count > 0 { try onChunk(vae.pcm[chunk.slot].floats(count: chunk.count)) }
            if stopProb > 0.5 && n > 3 {
                outcome.stopped = true
                break
            }
        }

        // The deferred latent and the last one, each in its own command buffer so the hidden
        // rows and pcm of one are read before the next overwrites them.
        while let d = vae.flushDecode() {
            let flushRec = StepRecorder(ctx: ctx, kernels: kernels, split: false)
            let e = flushRec.stage("flush")
            for dispatch in d.dispatches { dispatch(e) }
            _ = try flushRec.finish()
            if let capture { capture.vaeHidden.append(contentsOf: vaeHiddenRows()) }
            try onChunk(vae.pcm[d.chunk.slot].floats(count: d.chunk.count))
        }
        return outcome
    }

    /// One generation step after the prefill. On entry `z` holds the LLM output for this step
    /// when n == 0, else `xLLM` row 0 holds the aggregator output of the previous latent.
    /// The VAE frames of latent n-2 (the oldest with a lookahead latent) ride along with this
    /// step's LLM token, dispatch for dispatch on a concurrent encoder: the LLM GEMVs stream
    /// weights while the VAE's four-row GEMVs sit on the matrix unit, so the pair costs about
    /// the longer one. The chunk returned is theirs, one step later than the latent it decodes.
    private func encodeStep(_ n: Int, rec: StepRecorder, cfg: Float, useTeacher: Bool) -> (slot: Int, count: Int) {
        let hidden = llm.stack.config.hidden
        let h = DiTWeights.hidden
        let zView = ActView(z, ld: hidden)
        var e = rec.stage("llm", concurrent: true)
        let token = n > 0 ? llmForward.dispatches(x: ActView(xLLM, ld: hidden), rows: 1, normRows: 1, out: zView) : []
        let frames = vae.deferredDecode()
        e.encodePaired(token, frames?.dispatches ?? [])
        e.linear(zView, rows: 1, llm.stopHead, ActView(stopLogits, offset: (n % 2) * 8 * 4, ld: 8))

        e = rec.stage("flow")
        e.linear(zView, rows: 1, flowWeights.condEmbedder, ActView(cemb, ld: h))
        if n > 0 {
            let prev = useTeacher ? ActView(teacher, offset: (n - 1) * Self.latentDim * 4, ld: Self.latentDim) : ActView(y, ld: Self.latentDim)
            e.copy(prev, count: Self.latentDim, to: ActView(history, ld: Self.latentDim))
        }
        e.copy(ActView(noise, offset: n * Self.latentDim * 4, ld: Self.latentDim), count: Self.latentDim, to: ActView(y, ld: Self.latentDim))
        let xFlow = ActView(ditX, ld: h)
        for k in 0..<flowSteps {
            e.flowAssemble(y: y, history: history, cemb: cemb, condBias: flowWeights.condEmbedder.bias!,
                           temb: ActView(temb, offset: k * h * 4, ld: h), xEmbedder: flowWeights.xEmbedder, xFlow)
            flow.encode(e, x: xFlow, batch: 2, seq: 3, outRows: 6, out: ActView(ditOut, ld: Self.latentDim))
            e.flowEuler(y: y, pred: ActView(ditOut, ld: Self.latentDim), predRow: 2, nullRow: 5, latentDim: Self.latentDim,
                        cfg: cfg, dt: timesteps[k + 1] - timesteps[k],
                        capture: ActView(velocity, offset: n == 0 ? k * 2 * Self.latentDim * 4 : 0, ld: Self.latentDim))
        }
        e.copy(ActView(y, ld: Self.latentDim), count: Self.latentDim, to: ActView(latents, offset: n * Self.latentDim * 4, ld: Self.latentDim))
        let latent = useTeacher ? ActView(teacher, offset: n * Self.latentDim * 4, ld: Self.latentDim) : ActView(y, ld: Self.latentDim)

        e = rec.stage("agg")
        let xAgg = ActView(aggX, ld: h)
        e.copy(ActView(cls, ld: h), count: h, to: xAgg)
        e.linear(latent, rows: 1, aggWeights.xEmbedder, xAgg.row(1))
        agg.encode(e, x: xAgg, batch: 1, seq: 2, outRows: 1, out: ActView(xLLM, ld: hidden))

        e = rec.stage("vae")
        vae.encodeFc1(e, latent: latent)
        return frames?.chunk ?? (0, 0)
    }

    private func record(_ capture: ParityCapture, step n: Int, stopProb: Float, chunk: (slot: Int, count: Int)) {
        let hidden = llm.stack.config.hidden
        capture.llmHidden.append(z.floats(count: hidden))
        capture.stopProb.append(stopProb)
        capture.latents.append(latents.floats(count: Self.latentDim, offset: n * Self.latentDim * 4))
        capture.aggOut.append(xLLM.floats(count: hidden))
        if n == 0 { capture.velocityStep0 = velocity.floats(count: flowSteps * 2 * Self.latentDim) }
        if chunk.count > 0 { capture.vaeHidden.append(contentsOf: vaeHiddenRows()) }
    }

    private func vaeHiddenRows() -> [[Float]] {
        let hidden = vaeWeights.stack.config.hidden
        return (0..<4).map { vae.hidden.floats(count: hidden, offset: $0 * hidden * 4) }
    }

    /// Standard normal samples from a SplitMix64 stream, Box-Muller.
    static func gaussianNoise(count: Int, seed: UInt64) -> [Float] {
        var state = seed &+ 0x9E3779B97F4A7C15
        func next() -> Double {
            state &+= 0x9E3779B97F4A7C15
            var zz = state
            zz = (zz ^ (zz >> 30)) &* 0xBF58476D1CE4E5B9
            zz = (zz ^ (zz >> 27)) &* 0x94D049BB133111EB
            zz ^= zz >> 31
            return Double(zz >> 11) / Double(1 << 53)
        }
        var out = [Float](repeating: 0, count: count)
        var i = 0
        while i < count {
            let u1 = max(next(), 1e-300), u2 = next()
            let r = (-2 * log(u1)).squareRoot()
            out[i] = Float(r * cos(2 * Double.pi * u2))
            if i + 1 < count { out[i + 1] = Float(r * sin(2 * Double.pi * u2)) }
            i += 2
        }
        return out
    }
}
