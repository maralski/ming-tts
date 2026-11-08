// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// In-situ timing of one Qwen stack step: 24 synthetic layers carved from a 1 GiB bf16 pool (so
/// every layer streams fresh weights, like the model), the real `QwenForward.encode`, and a
/// per-kernel breakdown of the same layer sequence so a change in the whole can be attributed.
/// The LLM case is one token at cache position 230 (the bench text's mean), the VAE case is
/// four frames under the 65-key sliding window, the prefill case the bench text's 174 rows.
public enum QwenBench {
    public static func run(rounds: Int = 7) throws {
        let ctx = try MetalContext()
        let kernels = try Kernels(ctx: ctx)
        let pool = try ctx.makeBuffer(bytes: 1 << 30, label: "qbench.pool")
        GemvBench.fillBF16(pool)
        let cfg = QwenConfig()
        var vaeCfg = QwenConfig()
        vaeCfg.window = 65
        let h = cfg.hidden, inter = cfg.intermediate, qkvN = cfg.qkvWidth
        var offset = 0
        func slice(_ count: Int) -> GPUTensor {
            defer { offset += (count * 2 + 255) & ~255 }
            return GPUTensor(buffer: pool, offset: offset, count: count)
        }
        let layers = (0..<cfg.layers).map { _ in
            QwenLayerWeights(norm1: slice(h),
                             qkv: Linear(w: slice(qkvN * h), bias: slice(qkvN), n: qkvN, k: h),
                             o: Linear(w: slice(h * h), bias: nil, n: h, k: h),
                             norm2: slice(h),
                             gate: Linear(w: slice(inter * h), bias: nil, n: inter, k: h),
                             up: Linear(w: slice(inter * h), bias: nil, n: inter, k: h),
                             down: Linear(w: slice(h * inter), bias: nil, n: h, k: inter))
        }
        let finalNorm = slice(h)
        print(String(format: "%.0f MB of layer weights per step", Double(offset) / 1e6))
        let out = try ctx.makeBuffer(bytes: 192 * h * 4, label: "qbench.out")

        func time(concurrent: Bool = false, _ body: (Encoder) -> Void) throws -> Double {
            let cb = ctx.queue.makeCommandBuffer()!
            let ce = cb.makeComputeCommandEncoder(dispatchType: concurrent ? .concurrent : .serial)!
            body(Encoder(k: kernels, ce: ce))
            ce.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw MingTTSError.gpuFailure("\(e)") }
            return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
        }
        // Variants are timed interleaved, one command buffer each per round, so they share the
        // noise of other sessions on the GPU; two untimed rounds first let the GPU clock up from
        // idle. The minimum is a variant's own cost under contention, the median what a step sees.
        func stats(_ bodies: [(concurrent: Bool, body: (Encoder) -> Void)]) throws -> [(median: Double, min: Double)] {
            var v = [[Double]](repeating: [], count: bodies.count)
            for i in 0..<(rounds + 2) {
                for (b, variant) in bodies.enumerated() {
                    let t = try time(concurrent: variant.concurrent, variant.body)
                    if i >= 2 { v[b].append(t) }
                }
            }
            return v.map { let s = $0.sorted(); return (s[rounds / 2], s[0]) }
        }
        func report(_ title: String, _ variants: [(String, Bool, (Encoder) -> Void)]) throws {
            let st = try stats(variants.map { ($0.1, $0.2) })
            print(title)
            for ((label, _, _), s) in zip(variants, st) {
                print(String(format: "  %-26@ %6.3f ms (min %6.3f)", label as NSString, s.median, s.min))
            }
        }

        typealias Part = (Encoder, QwenLayerWeights, KVCacheLayer) -> Void
        /// One Qwen stack case with its own workspace and cache, and the layer sequence as parts.
        struct Stack {
            let name: String
            let rows, pos0: Int
            let cache: KVCache
            let forward: QwenForward
            let step: (Encoder) -> Void
            let parts: [(String, Part)]
            let layer: [Part]
            func overLayers(_ parts: [Part]) -> (Encoder) -> Void {
                return { e in
                    for (i, l) in forward.weights.layers.enumerated() { for part in parts { part(e, l, cache.layers[i]) } }
                }
            }
        }
        func makeStack(_ name: String, _ c: QwenConfig, rows: Int, pos0: Int) throws -> Stack {
            let work = try QwenWorkspace(ctx: ctx, config: c, maxRows: max(rows, 8), label: "qbench.\(name)")
            let cache = try KVCache(ctx: ctx, config: c, capacity: 512, label: "qbench.\(name).cache")
            let forward = try QwenForward(ctx: ctx, weights: QwenWeights(config: c, layers: layers, finalNorm: finalNorm), cache: cache, work: work)
            let xb = try ctx.makeBuffer(bytes: (rows + 8) * h * 4, label: "qbench.\(name).x")
            GemvBench.fillF32(xb)
            let xv = ActView(xb, ld: h), ov = ActView(out, ld: h)
            let normed = ActView(work.normed, ld: h), qkv = ActView(work.qkv, ld: qkvN)
            let attn = ActView(work.attn, ld: h), mlp = ActView(work.mlp, ld: inter)
            let rmsnorm1: Part = { e, l, _ in e.rmsnorm(xv, rows: rows, width: h, weight: l.norm1, normed, eps: c.eps) }
            let rmsnorm2: Part = { e, l, _ in e.rmsnorm(xv, rows: rows, width: h, weight: l.norm2, normed, eps: c.eps) }
            let qkvPlain: Part = { e, l, _ in e.linear(normed, rows: rows, l.qkv, qkv) }
            let rope: Part = { e, l, k in e.ropeCache(qkv, rows: rows, heads: c.heads, kvHeads: c.kvHeads, pos0: pos0, cache: k, invFreq: forward.invFreq) }
            let attention: Part = { e, l, k in e.attention(qkv, rows: rows, heads: c.heads, kvHeads: c.kvHeads, pos0: pos0, window: c.window, cache: k, attn) }
            let oRes: Part = { e, l, _ in e.linear(attn, rows: rows, l.o, xv, flags: .accumulate) }
            let gateUpPlain: Part = { e, l, _ in e.linear(normed, rows: rows, l.gate, mlp, flags: .siluGate, gate: l.up.w) }
            let downRes: Part = { e, l, _ in e.linear(mlp, rows: rows, l.down, xv, flags: .accumulate) }
            return Stack(name: name, rows: rows, pos0: pos0, cache: cache, forward: forward,
                         step: { e in
                             cache.reset()
                             cache.advance(pos0)
                             forward.encode(e, x: xv, rows: rows, normRows: rows, out: ov)
                         },
                         parts: [("rmsnorm", rmsnorm1), ("qkv", qkvPlain), ("rope_cache", rope), ("attention", attention),
                                 ("o+res", oRes), ("gate|up", gateUpPlain), ("down+res", downRes)],
                         layer: [rmsnorm1, qkvPlain, rope, attention, oRes, rmsnorm2, gateUpPlain, downRes])
        }

        let llm = try makeStack("llm", cfg, rows: 1, pos0: 230)
        let vae = try makeStack("vae", vaeCfg, rows: 4, pos0: 300)
        let prefill = try makeStack("prefill", cfg, rows: 174, pos0: 0)
        for s in [llm, vae, prefill] {
            // The real step, then the layer sequence assembled here from the parts below.
            try report(String(format: "%@ (M=%d, pos0 %d), %d dispatches in QwenForward.encode, median of %d:", s.name, s.rows, s.pos0, s.forward.dispatchesPerStep(rows: s.rows), rounds), [
                ("QwenForward.encode", false, s.step),
                ("layer sequence", false, s.overLayers(s.layer)),
            ])
            // Each kernel of the layer sequence on its own, over the 24 layers' weights.
            let partStats = try stats(s.parts.map { (false, s.overLayers([$0.1])) })
            let n = Double(layers.count)
            for ((label, _), st) in zip(s.parts, partStats) {
                print(String(format: "  %-14@ %7.1f us per dispatch (min %6.1f)  %6.3f ms per step", label as NSString, st.median * 1e3 / n, st.min * 1e3 / n, st.median))
            }
        }

        // The VAE step of latent n does not feed the LLM step of token n+1. Serial, one after the
        // other, against a concurrent encoder that pairs the two stacks' dispatches wave by wave
        // (one buffers barrier per pair): the LLM GEMVs stream weights, the VAE GEMVs sit on the
        // matrix unit, so a pair may finish in the time of the longer one.
        try report("llm + vae steps, median of \(rounds):", [
            ("serial, llm then vae", false, { e in llm.step(e); vae.step(e) }),
            ("concurrent, paired waves", true, { e in
                for (i, l) in layers.enumerated() {
                    for (a, b) in zip(llm.layer, vae.layer) {
                        a(e, l, llm.cache.layers[i])
                        b(e, l, vae.cache.layers[i])
                        e.ce.memoryBarrier(scope: .buffers)
                    }
                }
            }),
            ("concurrent, llm then vae", true, { e in
                for (i, l) in layers.enumerated() {
                    for a in llm.layer { a(e, l, llm.cache.layers[i]); e.ce.memoryBarrier(scope: .buffers) }
                }
                for (i, l) in layers.enumerated() {
                    for b in vae.layer { b(e, l, vae.cache.layers[i]); e.ce.memoryBarrier(scope: .buffers) }
                }
            }),
        ])
    }
}
