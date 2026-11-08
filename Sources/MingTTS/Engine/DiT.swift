// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Scratch for a DiT stack forward over at most `maxRows` rows (plus the 8 spare GEMV rows).
final class DiTWorkspace {
    let normed: MTLBuffer
    let qkv: MTLBuffer
    let attn: MTLBuffer
    let ff: MTLBuffer

    init(ctx: MetalContext, maxRows: Int, label: String) throws {
        let rows = maxRows + 8, h = DiTWeights.hidden
        normed = try ctx.makeBuffer(bytes: rows * h * 4, label: "\(label).normed")
        qkv = try ctx.makeBuffer(bytes: rows * 3 * h * 4, label: "\(label).qkv")
        attn = try ctx.makeBuffer(bytes: rows * h * 4, label: "\(label).attn")
        ff = try ctx.makeBuffer(bytes: rows * 4 * h * 4, label: "\(label).ff")
    }
}

/// Encodes the DiT block stack plus final layer. `x` holds `batch * seq` rows of the hidden
/// stream and is updated in place; the final layer of the first `outRows` rows lands in `out`.
struct DiTForward {
    let weights: DiTWeights
    let work: DiTWorkspace
    let invFreq: MTLBuffer

    init(ctx: MetalContext, weights: DiTWeights, work: DiTWorkspace) throws {
        self.weights = weights
        self.work = work
        invFreq = try ctx.makeBuffer(floats: weights.invFreq, label: "dit.invFreq")
    }

    func encode(_ e: Encoder, x: ActView, batch: Int, seq: Int, outRows: Int, out: ActView) {
        let h = DiTWeights.hidden, rows = batch * seq
        let normed = ActView(work.normed, ld: h)
        let qkv = ActView(work.qkv, ld: 3 * h)
        let attn = ActView(work.attn, ld: h)
        let ff = ActView(work.ff, ld: 4 * h)
        for block in weights.blocks {
            e.rmsnorm(x, rows: rows, width: h, weight: block.norm1, normed)
            e.linear(normed, rows: rows, block.qkv, qkv)
            e.ditAttention(qkv, batch: batch, seq: seq, heads: DiTWeights.heads, invFreq: invFreq, attn)
            e.linear(attn, rows: rows, block.out, x, flags: .accumulate)
            e.rmsnorm(x, rows: rows, width: h, weight: block.norm2, normed)
            e.linear(normed, rows: rows, block.ff1, ff, flags: .gelu)
            e.linear(ff, rows: rows, block.ff2, x, flags: .accumulate)
        }
        e.rmsnorm(x, rows: outRows, width: h, weight: weights.finalNorm, normed)
        e.linear(normed, rows: outRows, weights.finalLinear, out)
    }
}

/// The CFM sampler schedule: EPSS timesteps for 10 steps with sway coefficient -1.
enum FlowSchedule {
    static func timesteps(steps: Int) -> [Float] {
        let epss: [Int: [Float]] = [
            5: [0, 2, 4, 8, 16, 32], 6: [0, 2, 4, 6, 8, 16, 32], 7: [0, 2, 4, 6, 8, 16, 24, 32],
            10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], 12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
            16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        ]
        let base = epss[steps].map { $0.map { $0 / 32 } } ?? (0...steps).map { Float($0) / Float(steps) }
        return base.map { t in t - (cos(Float.pi / 2 * t) - 1 + t) }
    }

    /// SinusPositionEmbedding(256) at scale 1000, evaluated the way torch does in fp32.
    static func sinusoid(_ t: Float) -> [Float] {
        let half = 128
        let step = Float(log(10000.0) / Double(half - 1))
        var out = [Float](repeating: 0, count: 2 * half)
        for i in 0..<half {
            let freq = expf(Float(i) * -step)
            let arg = (1000 * t) * freq
            out[i] = Float(sin(Double(arg)))
            out[half + i] = Float(cos(Double(arg)))
        }
        return out
    }
}
