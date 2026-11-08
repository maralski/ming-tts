// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Per-layer K and V caches for one Qwen stack, fp32, one buffer for all layers.
final class KVCache {
    let layers: [KVCacheLayer]
    let capacity: Int
    private(set) var length = 0

    init(ctx: MetalContext, config: QwenConfig, capacity: Int, label: String) throws {
        self.capacity = capacity
        let perTensor = config.kvHeads * capacity * 64 * 4
        let aligned = (perTensor + 255) & ~255
        let buffer = try ctx.makeBuffer(bytes: aligned * 2 * config.layers, label: label)
        layers = (0..<config.layers).map { i in
            KVCacheLayer(k: GPUTensor(buffer: buffer, offset: (2 * i) * aligned, count: perTensor / 4),
                         v: GPUTensor(buffer: buffer, offset: (2 * i + 1) * aligned, count: perTensor / 4),
                         capacity: capacity)
        }
    }

    func reset() { length = 0 }
    func advance(_ rows: Int) { length += rows }
}

/// Scratch activations for a Qwen stack forward over at most `maxRows` rows. Every buffer has 8
/// spare rows past `maxRows` because the GEMV reads whole row chunks.
final class QwenWorkspace {
    let maxRows: Int
    let normed: MTLBuffer
    let qkv: MTLBuffer
    let attn: MTLBuffer
    let mlp: MTLBuffer

    init(ctx: MetalContext, config: QwenConfig, maxRows: Int, label: String) throws {
        self.maxRows = maxRows
        let rows = maxRows + 8
        normed = try ctx.makeBuffer(bytes: rows * config.hidden * 4, label: "\(label).normed")
        qkv = try ctx.makeBuffer(bytes: rows * config.qkvWidth * 4, label: "\(label).qkv")
        attn = try ctx.makeBuffer(bytes: rows * config.hidden * 4, label: "\(label).attn")
        mlp = try ctx.makeBuffer(bytes: rows * config.intermediate * 4, label: "\(label).mlp")
    }
}

/// Encodes a Qwen2 decoder stack: `x` holds `rows` embeddings on entry and the residual stream
/// after the last layer on exit; the final norm of `rows` rows goes to `out`.
struct QwenForward {
    let weights: QwenWeights
    let cache: KVCache
    let work: QwenWorkspace
    let invFreq: MTLBuffer

    init(ctx: MetalContext, weights: QwenWeights, cache: KVCache, work: QwenWorkspace) throws {
        self.weights = weights
        self.cache = cache
        self.work = work
        let theta = weights.config.ropeTheta
        invFreq = try ctx.makeBuffer(floats: (0..<32).map { 1 / powf(theta, Float(2 * $0) / 64) }, label: "invFreq")
    }

    /// Runs the layers over `rows` rows of `x` at positions cache.length..., then norms
    /// the rows `normRows` (counted from the end) into `out`.
    func encode(_ e: Encoder, x: ActView, rows: Int, normRows: Int, out: ActView, afterLayer: ((Int, ActView) -> Void)? = nil) {
        for d in dispatches(x: x, rows: rows, normRows: normRows, out: out, afterLayer: afterLayer) { d(e) }
    }

    /// The dispatches of `encode`, one closure each, in order, for a caller that interleaves
    /// them with another stack's (`Encoder.encodePaired`). Positions are taken from the cache
    /// now, so build and encode in the same order.
    func dispatches(x: ActView, rows: Int, normRows: Int, out: ActView, afterLayer: ((Int, ActView) -> Void)? = nil) -> [(Encoder) -> Void] {
        precondition(rows <= work.maxRows && cache.length + rows <= cache.capacity)
        let c = weights.config
        let normed = ActView(work.normed, ld: c.hidden)
        let qkv = ActView(work.qkv, ld: c.qkvWidth)
        let attn = ActView(work.attn, ld: c.hidden)
        let mlp = ActView(work.mlp, ld: c.intermediate)
        let pos0 = cache.length
        let invFreq = self.invFreq
        var list: [(Encoder) -> Void] = []
        for (i, layer) in weights.layers.enumerated() {
            let kv = cache.layers[i]
            list.append { $0.rmsnorm(x, rows: rows, width: c.hidden, weight: layer.norm1, normed, eps: c.eps) }
            list.append { $0.linear(normed, rows: rows, layer.qkv, qkv) }
            list.append { $0.ropeCache(qkv, rows: rows, heads: c.heads, kvHeads: c.kvHeads, pos0: pos0, cache: kv, invFreq: invFreq) }
            list.append { $0.attention(qkv, rows: rows, heads: c.heads, kvHeads: c.kvHeads, pos0: pos0, window: c.window, cache: kv, attn) }
            list.append { $0.linear(attn, rows: rows, layer.o, x, flags: .accumulate) }
            list.append { $0.rmsnorm(x, rows: rows, width: c.hidden, weight: layer.norm2, normed, eps: c.eps) }
            list.append { $0.linear(normed, rows: rows, layer.gate, mlp, flags: .siluGate, gate: layer.up.w) }
            list.append { $0.linear(mlp, rows: rows, layer.down, x, flags: .accumulate) }
            if let afterLayer { list.append { _ in afterLayer(i, x) } }
        }
        list.append { $0.rmsnorm(x.row(rows - normRows), rows: normRows, width: c.hidden, weight: weights.finalNorm, out, eps: c.eps) }
        cache.advance(rows)
        return list
    }

    /// Dispatches `encode` issues for `rows` rows, for the bench's per-dispatch arithmetic.
    func dispatchesPerStep(rows: Int) -> Int { weights.config.layers * 8 + 1 }
}

extension Encoder {
    /// Encodes two independent dispatch lists in lockstep on a concurrent encoder: one of
    /// each, then a barrier, so the pair shares the GPU while each list keeps its own order.
    /// Leftovers of the longer list follow one per barrier. On a serial encoder the barriers
    /// are redundant and the lists simply alternate.
    func encodePaired(_ a: [(Encoder) -> Void], _ b: [(Encoder) -> Void]) {
        for i in 0..<max(a.count, b.count) {
            if i < a.count { a[i](self) }
            if i < b.count { b[i](self) }
            ce.memoryBarrier(scope: .buffers)
        }
    }
}
