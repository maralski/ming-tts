// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// A bf16 (or fp32) tensor living in a GPU buffer at a byte offset.
struct GPUTensor {
    let buffer: MTLBuffer
    let offset: Int
    let count: Int
}

/// y = x W^T (+ bias). `w` is bf16 [n, k] row-major.
struct Linear {
    let w: GPUTensor
    let bias: GPUTensor?
    let n: Int
    let k: Int
}

struct GemvFlags: OptionSet {
    let rawValue: UInt32
    static let bias = GemvFlags(rawValue: 1)
    static let accumulate = GemvFlags(rawValue: 2)
    static let gelu = GemvFlags(rawValue: 4)
    static let siluGate = GemvFlags(rawValue: 8)
    static let silu = GemvFlags(rawValue: 16)
}

/// An fp32 activation matrix view: `rows` x `width` floats at `offset` bytes, row stride `ld`.
struct ActView {
    let buffer: MTLBuffer
    let offset: Int
    let ld: Int

    init(_ buffer: MTLBuffer, offset: Int = 0, ld: Int) {
        self.buffer = buffer
        self.offset = offset
        self.ld = ld
    }

    func row(_ r: Int) -> ActView { ActView(buffer, offset: offset + r * ld * 4, ld: ld) }
    func column(_ c: Int) -> ActView { ActView(buffer, offset: offset + c * 4, ld: ld) }
}

/// The compiled pipelines, resolved once.
final class Kernels: @unchecked Sendable {
    let ctx: MetalContext
    let gemv: [Int: MTLComputePipelineState]
    /// gemm8 variants by (K split, column tiles).
    struct G8: Hashable { let ks, nc: Int }
    static let gemm8Variants = [G8(ks: 4, nc: 1), G8(ks: 4, nc: 2), G8(ks: 4, nc: 4), G8(ks: 8, nc: 1), G8(ks: 8, nc: 2), G8(ks: 8, nc: 4), G8(ks: 16, nc: 1), G8(ks: 16, nc: 2)]
    let gemm8: [G8: MTLComputePipelineState]
    let gemm: MTLComputePipelineState
    let rmsnorm, copyRows, ropeCache, ditAttention, flowAssemble, flowEuler: MTLComputePipelineState
    /// Attention pipelines by simdgroups per (row, head) item.
    let attention: [Int: MTLComputePipelineState]
    let upsample4, istftFrames, istftEmit: MTLComputePipelineState

    init(ctx: MetalContext) throws {
        self.ctx = ctx
        var g: [Int: MTLComputePipelineState] = [:]
        for mr in [1, 2] { g[mr] = try ctx.pipeline("gemv_\(mr)") }
        gemv = g
        var g8: [G8: MTLComputePipelineState] = [:]
        for v in Kernels.gemm8Variants { g8[v] = try ctx.pipeline("gemm8_\(v.ks)_\(v.nc)") }
        gemm8 = g8
        gemm = try ctx.pipeline("gemm_32x64x32")
        rmsnorm = try ctx.pipeline("rmsnorm")
        copyRows = try ctx.pipeline("copy_rows")
        ropeCache = try ctx.pipeline("rope_cache")
        attention = [1: try ctx.pipeline("attention_cache_1"), 4: try ctx.pipeline("attention_cache_4")]
        ditAttention = try ctx.pipeline("dit_attention")
        flowAssemble = try ctx.pipeline("flow_assemble")
        flowEuler = try ctx.pipeline("flow_euler")
        upsample4 = try ctx.pipeline("upsample4")
        istftFrames = try ctx.pipeline("istft_frames")
        istftEmit = try ctx.pipeline("istft_emit")
    }

    /// The tiled GEMM takes over from the GEMV at this many rows (prefill), when the shape tiles.
    static let gemmMinRows = 16
    static let gemmBM = 32, gemmBN = 64, gemmBK = 32
    static func usesGemm(rows m: Int, n: Int, k: Int) -> Bool {
        m >= gemmMinRows && n % gemmBN == 0 && k % gemmBK == 0
    }
}

/// Encodes kernels onto one compute encoder. Every method appends a dispatch and nothing else.
struct Encoder {
    let k: Kernels
    let ce: MTLComputeCommandEncoder

    private func setBytes<T>(_ value: T, index: Int) {
        withUnsafeBytes(of: value) { ce.setBytes($0.baseAddress!, length: $0.count, index: index) }
    }

    /// Simdgroups splitting K in one gemm8 threadgroup. Eight (256 threads) keeps every threadgroup
    /// of the decode shapes resident in one wave; 16 made 512-thread groups that ran in two waves on
    /// small N. Overridable for the micro-bench.
    nonisolated(unsafe) static var kSplitOverride: Int?
    static func kSplit(forK k: Int) -> Int {
        if let s = kSplitOverride { return s }
        for ks in [8, 4] where k % (8 * ks) == 0 { return ks }
        preconditionFailure("gemm8 needs K % 32 == 0, got \(k)")
    }

    /// Column tiles per gemm8 simdgroup: a threadgroup covers 8 nc columns and each simdgroup streams
    /// nc weight rows against one X load. Measured rule (.audit/hillclimb-xload.tsv h3, h3b, a1): two
    /// tiles cut the K=4096 and K=4864 decode projections by 16-22% per dispatch and are within noise
    /// or worse on every K <= 1024 shape, so the threshold separates those two groups. X traffic is not
    /// the reason: removing the X loads outright saved 2% on the same shapes. The long-K shapes are the
    /// ones a serial-vs-concurrent probe showed latency-bound, and two tiles give each lane two
    /// independent weight streams with half the threadgroups, which fits but is unverified. Four tiles
    /// lose on every decode shape. The kernel needs N % (8 nc) == 0 above one tile. Overridable for
    /// the micro-bench.
    nonisolated(unsafe) static var columnTilesOverride: Int?
    static func columnTiles(forN n: Int, k: Int) -> Int {
        let want = columnTilesOverride ?? (k > 1024 ? 2 : 1)
        for nc in [want, 2, 1] where nc <= 4 && n % (8 * nc) == 0 { return nc }
        return 1
    }

    /// Row counts up to this take the scalar kernel; the matrix unit pads rows to 8, which costs
    /// more than one or two scalar FMAs per weight. Overridable for the micro-bench.
    nonisolated(unsafe) static var scalarRowsMax = 2

    /// y[m, n] = sum_k x[m, k] w[n, k] with the epilogue in `flags`. `gate` holds the second
    /// matrix for `.siluGate`, in which case `linear` is the gate projection.
    /// Rows of x past m are never read. Prefill shapes (16 rows and up) go to the tiled GEMM,
    /// one or two rows to the scalar kernel, anything between to the matrix-unit kernel.
    func linear(_ x: ActView, rows m: Int, _ linear: Linear, _ y: ActView, flags: GemvFlags = [], gate: GPUTensor? = nil) {
        struct P { var M, N, K, ldx, ldy, flags: UInt32; var eps: Float }
        var f = flags
        if linear.bias != nil { f.insert(.bias) }
        let groups: MTLSize, threads: Int
        if Kernels.usesGemm(rows: m, n: linear.n, k: linear.k) {
            ce.setComputePipelineState(k.gemm)
            groups = MTLSize(width: linear.n / Kernels.gemmBN, height: (m + Kernels.gemmBM - 1) / Kernels.gemmBM, depth: 1)
            threads = 128
        } else if m <= Encoder.scalarRowsMax {
            ce.setComputePipelineState(k.gemv[m]!)
            groups = MTLSize(width: (linear.n + 3) / 4, height: 1, depth: 1)
            threads = 128
        } else {
            let ks = Encoder.kSplit(forK: linear.k), nc = Encoder.columnTiles(forN: linear.n, k: linear.k)
            guard let pipeline = k.gemm8[Kernels.G8(ks: ks, nc: nc)] else { preconditionFailure("gemm8 has no (ks \(ks), nc \(nc)) variant") }
            ce.setComputePipelineState(pipeline)
            groups = MTLSize(width: (linear.n + 8 * nc - 1) / (8 * nc), height: (m + 7) / 8, depth: 1)
            threads = 32 * ks
        }
        ce.setBuffer(x.buffer, offset: x.offset, index: 0)
        ce.setBuffer(linear.w.buffer, offset: linear.w.offset, index: 1)
        let w2 = gate ?? linear.w
        ce.setBuffer(w2.buffer, offset: w2.offset, index: 2)
        let b = linear.bias ?? linear.w
        ce.setBuffer(b.buffer, offset: b.offset, index: 3)
        ce.setBuffer(y.buffer, offset: y.offset, index: 4)
        setBytes(P(M: UInt32(m), N: UInt32(linear.n), K: UInt32(linear.k), ldx: UInt32(x.ld), ldy: UInt32(y.ld), flags: f.rawValue, eps: 0), index: 5)
        ce.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    func rmsnorm(_ x: ActView, rows: Int, width: Int, weight: GPUTensor, _ y: ActView, eps: Float = 1e-6) {
        struct P { var K, ldx, ldy: UInt32; var eps: Float }
        ce.setComputePipelineState(k.rmsnorm)
        ce.setBuffer(x.buffer, offset: x.offset, index: 0)
        ce.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        ce.setBuffer(y.buffer, offset: y.offset, index: 2)
        setBytes(P(K: UInt32(width), ldx: UInt32(x.ld), ldy: UInt32(y.ld), eps: eps), index: 3)
        ce.dispatchThreads(MTLSize(width: rows * 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    func copy(_ x: ActView, count: Int, to y: ActView) {
        ce.setComputePipelineState(k.copyRows)
        ce.setBuffer(x.buffer, offset: x.offset, index: 0)
        ce.setBuffer(y.buffer, offset: y.offset, index: 1)
        ce.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(count, 256), height: 1, depth: 1))
    }

    /// Applies NeoX RoPE to the q heads of `qkv` in place and appends k (rotated) and v to the cache
    /// at positions pos0..<pos0+m.
    func ropeCache(_ qkv: ActView, rows m: Int, heads hq: Int, kvHeads hkv: Int, pos0: Int, cache: KVCacheLayer, invFreq: MTLBuffer) {
        struct P { var M, hq, hkv, pos0, ldq, cacheLen: UInt32 }
        ce.setComputePipelineState(k.ropeCache)
        ce.setBuffer(qkv.buffer, offset: qkv.offset, index: 0)
        ce.setBuffer(cache.k.buffer, offset: cache.k.offset, index: 1)
        ce.setBuffer(cache.v.buffer, offset: cache.v.offset, index: 2)
        ce.setBuffer(invFreq, offset: 0, index: 3)
        setBytes(P(M: UInt32(m), hq: UInt32(hq), hkv: UInt32(hkv), pos0: UInt32(pos0), ldq: UInt32(qkv.ld), cacheLen: UInt32(cache.capacity)), index: 4)
        ce.dispatchThreads(MTLSize(width: 32, height: hq + 2 * hkv, depth: m), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// Causal attention of the m query rows at positions pos0..<pos0+m over the cache. Decode
    /// rows spread each (row, head) over four simdgroups; prefill rows have items enough.
    func attention(_ q: ActView, rows m: Int, heads hq: Int, kvHeads hkv: Int, pos0: Int, window: Int, cache: KVCacheLayer, _ out: ActView) {
        struct P { var M, hq, hkv, pos0, window, ldq, ldo, cacheLen: UInt32; var scale: Float }
        let split = m < Kernels.gemmMinRows ? 4 : 1
        ce.setComputePipelineState(k.attention[split]!)
        ce.setBuffer(q.buffer, offset: q.offset, index: 0)
        ce.setBuffer(cache.k.buffer, offset: cache.k.offset, index: 1)
        ce.setBuffer(cache.v.buffer, offset: cache.v.offset, index: 2)
        ce.setBuffer(out.buffer, offset: out.offset, index: 3)
        setBytes(P(M: UInt32(m), hq: UInt32(hq), hkv: UInt32(hkv), pos0: UInt32(pos0), window: UInt32(window), ldq: UInt32(q.ld), ldo: UInt32(out.ld), cacheLen: UInt32(cache.capacity), scale: 1 / Float(64).squareRoot()), index: 4)
        let itemsPerGroup = 4 / split
        ce.dispatchThreadgroups(MTLSize(width: (m * hq + itemsPerGroup - 1) / itemsPerGroup, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Full attention with interleaved RoPE over `batch` sequences of `seq` rows (the DiT); one
    /// simdgroup per (sequence, row, head), four per threadgroup.
    func ditAttention(_ qkv: ActView, batch: Int, seq: Int, heads: Int, invFreq: MTLBuffer, _ out: ActView) {
        struct P { var B, S, H, ld, ldo: UInt32; var scale: Float }
        ce.setComputePipelineState(k.ditAttention)
        ce.setBuffer(qkv.buffer, offset: qkv.offset, index: 0)
        ce.setBuffer(invFreq, offset: 0, index: 1)
        ce.setBuffer(out.buffer, offset: out.offset, index: 2)
        setBytes(P(B: UInt32(batch), S: UInt32(seq), H: UInt32(heads), ld: UInt32(qkv.ld), ldo: UInt32(out.ld), scale: 1 / Float(64).squareRoot()), index: 3)
        precondition(seq <= 8, "dit_attention holds at most 8 scores per lane")
        let work = batch * seq * heads
        ce.dispatchThreadgroups(MTLSize(width: (work + 3) / 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    func flowAssemble(y: MTLBuffer, history: MTLBuffer, cemb: MTLBuffer, condBias: GPUTensor, temb: ActView, xEmbedder: Linear, _ x: ActView) {
        struct P { var hidden, latentDim: UInt32 }
        ce.setComputePipelineState(k.flowAssemble)
        ce.setBuffer(y, offset: 0, index: 0)
        ce.setBuffer(history, offset: 0, index: 1)
        ce.setBuffer(cemb, offset: 0, index: 2)
        ce.setBuffer(condBias.buffer, offset: condBias.offset, index: 3)
        ce.setBuffer(temb.buffer, offset: temb.offset, index: 4)
        ce.setBuffer(xEmbedder.w.buffer, offset: xEmbedder.w.offset, index: 5)
        ce.setBuffer(xEmbedder.bias!.buffer, offset: xEmbedder.bias!.offset, index: 6)
        ce.setBuffer(x.buffer, offset: x.offset, index: 7)
        setBytes(P(hidden: UInt32(xEmbedder.n), latentDim: UInt32(xEmbedder.k)), index: 8)
        ce.dispatchThreads(MTLSize(width: xEmbedder.n, height: 6, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func flowEuler(y: MTLBuffer, pred: ActView, predRow: Int, nullRow: Int, latentDim: Int, cfg: Float, dt: Float, capture: ActView) {
        struct P { var latentDim, predRow, nullRow, ldp: UInt32; var cfg, dt: Float }
        ce.setComputePipelineState(k.flowEuler)
        ce.setBuffer(y, offset: 0, index: 0)
        ce.setBuffer(pred.buffer, offset: pred.offset, index: 1)
        ce.setBuffer(capture.buffer, offset: capture.offset, index: 2)
        setBytes(P(latentDim: UInt32(latentDim), predRow: UInt32(predRow), nullRow: UInt32(nullRow), ldp: UInt32(pred.ld), cfg: cfg, dt: dt), index: 3)
        ce.dispatchThreads(MTLSize(width: latentDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: latentDim, height: 1, depth: 1))
    }

    func upsample4(prev: ActView?, cur: ActView, next: ActView?, hidden: Int, _ frames: ActView) {
        struct P { var hidden, hasPrev, hasNext: UInt32 }
        ce.setComputePipelineState(k.upsample4)
        let p = prev ?? cur, n = next ?? cur
        ce.setBuffer(p.buffer, offset: p.offset, index: 0)
        ce.setBuffer(cur.buffer, offset: cur.offset, index: 1)
        ce.setBuffer(n.buffer, offset: n.offset, index: 2)
        ce.setBuffer(frames.buffer, offset: frames.offset, index: 3)
        setBytes(P(hidden: UInt32(hidden), hasPrev: prev == nil ? 0 : 1, hasNext: next == nil ? 0 : 1), index: 4)
        ce.dispatchThreads(MTLSize(width: hidden, height: 4, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// Synthesises `count` windowed frames from spectrum rows into ring slots starting at `frame0`.
    func istftFrames(spec: ActView, count: Int, nfft: Int, cosTable: MTLBuffer, window: MTLBuffer, frames: MTLBuffer, frame0: Int, ring: Int) {
        struct P { var nfft, ldSpec, frame0, ring: UInt32 }
        ce.setComputePipelineState(k.istftFrames)
        ce.setBuffer(spec.buffer, offset: spec.offset, index: 0)
        ce.setBuffer(cosTable, offset: 0, index: 1)
        ce.setBuffer(window, offset: 0, index: 2)
        ce.setBuffer(frames, offset: 0, index: 3)
        setBytes(P(nfft: UInt32(nfft), ldSpec: UInt32(spec.ld), frame0: UInt32(frame0), ring: UInt32(ring)), index: 4)
        ce.dispatchThreadgroups(MTLSize(width: (nfft + 255) / 256, height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Overlap-adds samples s0..<s0+count of the stream from the frames decoded so far.
    func istftEmit(frames: MTLBuffer, window: MTLBuffer, nfft: Int, hop: Int, s0: Int, count: Int, lastFrame: Int, ring: Int, _ out: ActView) {
        struct P { var nfft, hop, pad, s0, count, lastFrame, ring: UInt32 }
        guard count > 0 else { return }
        ce.setComputePipelineState(k.istftEmit)
        ce.setBuffer(frames, offset: 0, index: 0)
        ce.setBuffer(window, offset: 0, index: 1)
        ce.setBuffer(out.buffer, offset: out.offset, index: 2)
        setBytes(P(nfft: UInt32(nfft), hop: UInt32(hop), pad: UInt32((nfft - hop) / 2), s0: UInt32(s0), count: UInt32(count), lastFrame: UInt32(lastFrame), ring: UInt32(ring)), index: 3)
        ce.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}

/// One layer's K and V cache: fp32 [kvHeads, capacity, 64] each.
struct KVCacheLayer {
    let k: GPUTensor
    let v: GPUTensor
    let capacity: Int
}
