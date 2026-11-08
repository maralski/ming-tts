// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// The audio VAE encoder on Metal: `Encoder.forward` of the reference audio tokenizer, then the
/// posterior statistics. A 44.1 kHz waveform (a whole number of 3528-sample patches) becomes one
/// 64-dim mean and std per patch; the caller draws the sample.
final class VAEEncoder {
    /// Longest clip the buffers hold: 1600 frames is 32 s of audio.
    static let maxFrames = 1600
    static var maxLatents: Int { maxFrames / VAEEncoderWeights.patch }

    let ctx: MetalContext
    let kernels: Kernels
    let weights: VAEEncoderWeights
    let stackForward: QwenForward
    let aggregatorForward: QwenForward
    let frames: MTLBuffer
    let projected: MTLBuffer
    let x: MTLBuffer
    let normed: MTLBuffer
    let patches: MTLBuffer
    let patchesNormed: MTLBuffer
    let clsRows: MTLBuffer
    let cls: MTLBuffer
    let posterior: MTLBuffer

    init(ctx: MetalContext, kernels: Kernels, weights: VAEEncoderWeights) throws {
        self.ctx = ctx
        self.kernels = kernels
        self.weights = weights
        let h = VAEEncoderWeights.hidden
        let frameRows = Self.maxFrames + 8
        let patchRows = Self.maxLatents * (VAEEncoderWeights.patch + 1) + 8
        let stackCache = try KVCache(ctx: ctx, config: weights.stack.config, capacity: frameRows, label: "vae.enc.cache")
        stackForward = try QwenForward(ctx: ctx, weights: weights.stack, cache: stackCache,
                                       work: QwenWorkspace(ctx: ctx, config: weights.stack.config, maxRows: Self.maxFrames, label: "vae.enc"))
        let aggCache = try KVCache(ctx: ctx, config: weights.aggregator.config, capacity: patchRows, label: "vae.agg.cache")
        aggregatorForward = try QwenForward(ctx: ctx, weights: weights.aggregator, cache: aggCache,
                                            work: QwenWorkspace(ctx: ctx, config: weights.aggregator.config, maxRows: patchRows - 8, label: "vae.agg"))
        frames = try ctx.makeBuffer(bytes: frameRows * VAEEncoderWeights.paddedFrame * 4, label: "vae.enc.frames")
        projected = try ctx.makeBuffer(bytes: frameRows * h * 4, label: "vae.enc.projected")
        x = try ctx.makeBuffer(bytes: frameRows * h * 4, label: "vae.enc.x")
        normed = try ctx.makeBuffer(bytes: frameRows * h * 4, label: "vae.enc.normed")
        patches = try ctx.makeBuffer(bytes: patchRows * h * 4, label: "vae.enc.patches")
        patchesNormed = try ctx.makeBuffer(bytes: patchRows * h * 4, label: "vae.enc.patchesNormed")
        clsRows = try ctx.makeBuffer(bytes: (Self.maxLatents + 8) * h * 4, label: "vae.enc.clsRows")
        cls = try ctx.makeBuffer(floats: weights.cls, label: "vae.enc.cls")
        posterior = try ctx.makeBuffer(bytes: (Self.maxLatents + 8) * 2 * MingEngine.latentDim * 4, label: "vae.enc.posterior")
    }

    /// Posterior mean and std (`softplus(scale) + 1e-4`, the Oobleck distribution), one row of 64
    /// each per 3528-sample patch of `samples`. The samples are rounded to bf16 first, since the
    /// reference feeds the encoder a bf16 waveform.
    func encode(samples: [Float]) throws -> (mean: [Float], std: [Float]) {
        let frame = VAEEncoderWeights.frame, patch = VAEEncoderWeights.patch, h = VAEEncoderWeights.hidden
        guard samples.count % (frame * patch) == 0, !samples.isEmpty else {
            throw MingTTSError.badInput("reference audio must be a whole number of \(frame * patch)-sample patches, got \(samples.count)")
        }
        let frameCount = samples.count / frame
        guard frameCount <= Self.maxFrames else {
            throw MingTTSError.badInput("reference audio is \(frameCount) frames, the encoder holds \(Self.maxFrames) (\(Self.maxFrames * frame / 44100) s)")
        }
        let latents = frameCount / patch
        let rows = latents * (patch + 1)
        let width = VAEEncoderWeights.paddedFrame
        var padded = [Float](repeating: 0, count: frameCount * width)
        for f in 0..<frameCount {
            for i in 0..<frame { padded[f * width + i] = bf16Rounded(samples[f * frame + i]) }
        }
        frames.write(padded)
        stackForward.cache.reset()
        aggregatorForward.cache.reset()

        let rec = StepRecorder(ctx: ctx, kernels: kernels, split: false)
        let e = rec.stage("vae.encode")
        let framesView = ActView(frames, ld: width), projectedView = ActView(projected, ld: h)
        let xView = ActView(x, ld: h), normedView = ActView(normed, ld: h)
        e.linear(framesView, rows: frameCount, weights.fc1, projectedView)
        e.linear(projectedView, rows: frameCount, weights.fc2, xView)
        stackForward.encode(e, x: xView, rows: frameCount, normRows: frameCount, out: normedView)
        let patchesView = ActView(patches, ld: h), clsView = ActView(cls, ld: h)
        for p in 0..<latents {
            e.copy(normedView.row(p * patch), count: patch * h, to: patchesView.row(p * (patch + 1)))
            e.copy(clsView, count: h, to: patchesView.row(p * (patch + 1) + patch))
        }
        let patchesNormedView = ActView(patchesNormed, ld: h), clsRowsView = ActView(clsRows, ld: h)
        aggregatorForward.encode(e, x: patchesView, rows: rows, normRows: rows, out: patchesNormedView)
        for p in 0..<latents {
            e.copy(patchesNormedView.row(p * (patch + 1) + patch), count: h, to: clsRowsView.row(p))
        }
        e.linear(clsRowsView, rows: latents, weights.fc3, ActView(posterior, ld: 2 * MingEngine.latentDim))
        _ = try rec.finish()

        let dim = MingEngine.latentDim
        let raw = posterior.floats(count: latents * 2 * dim)
        var mean = [Float](repeating: 0, count: latents * dim), std = mean
        for p in 0..<latents {
            for i in 0..<dim {
                mean[p * dim + i] = raw[p * 2 * dim + i]
                std[p * dim + i] = softplus(raw[p * 2 * dim + dim + i]) + 1e-4
            }
        }
        return (mean, std)
    }
}

/// Round-to-nearest-even to bf16 and back, what `tensor.to(torch.bfloat16)` does to a sample.
func bf16Rounded(_ x: Float) -> Float {
    let u = x.bitPattern
    let rounded = (u &+ 0x7FFF &+ ((u >> 16) & 1)) & 0xFFFF_0000
    return Float(bitPattern: rounded)
}

/// `torch.nn.functional.softplus` with its default threshold of 20.
private func softplus(_ x: Float) -> Float { x > 20 ? x : log1p(exp(x)) }
