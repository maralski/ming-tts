// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Streaming audio VAE decoder. Each latent becomes fc1 output h[n]; once h[n+1] exists the four
/// frames of latent n are upsampled, run through the sliding-window Qwen stack, projected to a
/// spectrum, synthesised by the ISTFT and overlap-added. Samples are emitted as soon as every frame
/// that overlaps them has been decoded, which is what the reference stream does.
final class VAEDecoder {
    static let nfft = VAEDecoderWeights.nfft
    static let hop = VAEDecoderWeights.hop
    static let pad = (nfft - hop) / 2
    static let frameRing = 8
    /// Samples the last call can emit: four hops plus the trailing pad.
    static let maxChunk = 4 * hop + pad

    let weights: VAEDecoderWeights
    let forward: QwenForward
    let cache: KVCache
    let h: MTLBuffer
    let frames: MTLBuffer
    let hidden: MTLBuffer
    let spec: MTLBuffer
    let frameRing: MTLBuffer
    let cosTable: MTLBuffer
    let window: MTLBuffer
    let pcm: [MTLBuffer]

    /// Latents received so far (fc1 applied).
    private(set) var latents = 0
    /// Latents whose frames have been decoded.
    private(set) var decoded = 0
    /// Samples emitted so far.
    private(set) var emitted = 0

    init(ctx: MetalContext, weights: VAEDecoderWeights, maxLatents: Int) throws {
        self.weights = weights
        let cfg = weights.stack.config
        cache = try KVCache(ctx: ctx, config: cfg, capacity: 4 * maxLatents + 8, label: "vae.cache")
        let work = try QwenWorkspace(ctx: ctx, config: cfg, maxRows: 4, label: "vae")
        forward = try QwenForward(ctx: ctx, weights: weights.stack, cache: cache, work: work)
        h = try ctx.makeBuffer(bytes: 3 * cfg.hidden * 4, label: "vae.h")
        frames = try ctx.makeBuffer(bytes: 12 * cfg.hidden * 4, label: "vae.frames")
        hidden = try ctx.makeBuffer(bytes: 12 * cfg.hidden * 4, label: "vae.hidden")
        spec = try ctx.makeBuffer(bytes: 12 * weights.head.n * 4, label: "vae.spec")
        frameRing = try ctx.makeBuffer(bytes: Self.frameRing * Self.nfft * 4, label: "vae.frameRing")
        cosTable = try ctx.makeBuffer(floats: (0..<Self.nfft).map { Float(cos(2 * Double.pi * Double($0) / Double(Self.nfft))) }, label: "vae.cos")
        window = try ctx.makeBuffer(floats: weights.window, label: "vae.window")
        pcm = try (0..<2).map { try ctx.makeBuffer(bytes: Self.maxChunk * 4, label: "vae.pcm\($0)") }
    }

    func reset() {
        cache.reset()
        latents = 0
        decoded = 0
        emitted = 0
    }

    private func hRow(_ n: Int) -> ActView { ActView(h, offset: (n % 3) * weights.stack.config.hidden * 4, ld: weights.stack.config.hidden) }

    /// Feeds `latent` (64 floats in a buffer) as latent number `latents`: fc1 only. Its frames
    /// are decoded once the next latent exists (`deferredDecode`) or at the end (`flushDecode`).
    func encodeFc1(_ e: Encoder, latent: ActView) {
        e.linear(latent, rows: 1, weights.fc1, hRow(latents))
        latents += 1
    }

    /// The frames of the oldest undecoded latent that already has its lookahead latent, as
    /// dispatches for the caller to interleave with another stack's, and the pcm slot and
    /// sample count they emit. Nil when no such latent exists yet. One call per command buffer:
    /// the frame, hidden and spectrum buffers are single.
    func deferredDecode() -> (dispatches: [(Encoder) -> Void], chunk: (slot: Int, count: Int))? {
        guard decoded + 1 < latents else { return nil }
        return decodeDispatches(latentIndex: decoded, next: hRow(decoded + 1))
    }

    /// After the last latent: the remaining latents in order, the last without lookahead.
    /// Nil once everything is decoded. One call per command buffer, like `deferredDecode`.
    func flushDecode() -> (dispatches: [(Encoder) -> Void], chunk: (slot: Int, count: Int))? {
        guard decoded < latents else { return nil }
        let n = decoded
        return decodeDispatches(latentIndex: n, next: n + 1 < latents ? hRow(n + 1) : nil)
    }

    private func decodeDispatches(latentIndex n: Int, next: ActView?) -> (dispatches: [(Encoder) -> Void], chunk: (slot: Int, count: Int)) {
        let hid = weights.stack.config.hidden
        let framesView = ActView(frames, ld: hid)
        let prev: ActView? = n >= 1 ? hRow(n - 1) : nil
        let cur = hRow(n)
        var list: [(Encoder) -> Void] = [{ $0.upsample4(prev: prev, cur: cur, next: next, hidden: hid, framesView) }]
        list += forward.dispatches(x: framesView, rows: 4, normRows: 4, out: ActView(hidden, ld: hid))
        let specView = ActView(spec, ld: weights.head.n)
        let hiddenView = ActView(hidden, ld: hid), head = weights.head
        list.append { $0.linear(hiddenView, rows: 4, head, specView) }
        let cosTable = self.cosTable, window = self.window, frameRing = self.frameRing
        list.append { $0.istftFrames(spec: specView, count: 4, nfft: Self.nfft, cosTable: cosTable, window: window, frames: frameRing, frame0: 4 * n, ring: Self.frameRing) }
        let frameCount = 4 * (n + 1)
        let finalEnd = next == nil ? frameCount * Self.hop : frameCount * Self.hop - Self.pad
        let count = finalEnd - emitted
        let slot = n % 2
        let s0 = emitted, out = ActView(pcm[slot], ld: 1)
        list.append { $0.istftEmit(frames: frameRing, window: window, nfft: Self.nfft, hop: Self.hop, s0: s0, count: count, lastFrame: frameCount - 1, ring: Self.frameRing, out) }
        emitted = finalEnd
        decoded = n + 1
        return (list, (slot, count))
    }
}
