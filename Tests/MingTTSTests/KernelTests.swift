// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import MingTTS

/// Each kernel against a CPU reference on random data. bf16 weights are rounded on the host the
/// same way the GPU reads them, so the comparison isolates kernel arithmetic.
final class KernelTests: XCTestCase {
    static let ctx = try! MetalContext()
    static let kernels = try! Kernels(ctx: ctx)
    var ctx: MetalContext { Self.ctx }
    var kernels: Kernels { Self.kernels }

    func testShadersCompile() throws {
        try ctx.warmUpPipelines()
    }

    // MARK: helpers

    struct RNG {
        var state: UInt64
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / Float(1 << 24) * 2 - 1
        }
        mutating func floats(_ n: Int, scale: Float = 1) -> [Float] { (0..<n).map { _ in next() * scale } }
    }

    static func bf16Bits(_ x: Float) -> UInt16 {
        let u = x.bitPattern
        let rounded = u &+ 0x7FFF &+ ((u >> 16) & 1)
        return UInt16(rounded >> 16)
    }

    static func bf16Round(_ x: Float) -> Float { Float(bitPattern: UInt32(bf16Bits(x)) << 16) }

    func bf16Tensor(_ values: [Float], label: String) throws -> (GPUTensor, [Float]) {
        let bits = values.map(Self.bf16Bits)
        let b = try ctx.makeBuffer(bf16: bits, label: label)
        return (GPUTensor(buffer: b, offset: 0, count: values.count), values.map(Self.bf16Round))
    }

    func f32Buffer(_ values: [Float], label: String) throws -> MTLBuffer {
        try ctx.makeBuffer(floats: values, label: label)
    }

    func run(_ body: (Encoder) -> Void) throws {
        let cb = ctx.queue.makeCommandBuffer()!
        let ce = cb.makeComputeCommandEncoder()!
        body(Encoder(k: kernels, ce: ce))
        ce.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw MingTTSError.gpuFailure("\(e)") }
    }

    func assertClose(_ got: [Float], _ want: [Float], atol: Float, rtol: Float = 1e-4, _ what: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.count, want.count, what, file: file, line: line)
        var worst: (Int, Float, Float, Float) = (0, 0, 0, 0)
        for i in 0..<min(got.count, want.count) {
            let d = abs(got[i] - want[i])
            let tol = atol + rtol * abs(want[i])
            if d - tol > worst.1 - (atol + rtol * abs(worst.3)) { worst = (i, d, got[i], want[i]) }
        }
        let tol = atol + rtol * abs(worst.3)
        XCTAssertLessThanOrEqual(worst.1, tol, "\(what): index \(worst.0) got \(worst.2) want \(worst.3)", file: file, line: line)
    }

    // MARK: gemv

    func gemvCase(m: Int, n: Int, k: Int, flags: GemvFlags, bias: Bool, xScale: Float = 1) throws {
        var rng = RNG(state: UInt64(m * 1000 + n * 7 + k))
        // Rows past m hold NaN: the kernel must never read them.
        let x = rng.floats(m * k, scale: xScale) + [Float](repeating: .nan, count: 8 * k)
        let (w, wf) = try bf16Tensor(rng.floats(n * k, scale: 0.1), label: "w")
        let (w2, w2f) = try bf16Tensor(rng.floats(n * k, scale: 0.1), label: "w2")
        let (b, bf) = try bf16Tensor(rng.floats(n), label: "b")
        let y0 = rng.floats(m * n)
        let xb = try f32Buffer(x, label: "x")
        let yb = try f32Buffer(y0, label: "y")
        let lin = Linear(w: w, bias: bias ? b : nil, n: n, k: k)
        try run { e in
            e.linear(ActView(xb, ld: k), rows: m, lin, ActView(yb, ld: n), flags: flags, gate: flags.contains(.siluGate) ? w2 : nil)
        }
        var want = [Float](repeating: 0, count: m * n)
        for r in 0..<m {
            for c in 0..<n {
                var acc: Float = 0, acc2: Float = 0
                for i in 0..<k {
                    acc += x[r * k + i] * wf[c * k + i]
                    acc2 += x[r * k + i] * w2f[c * k + i]
                }
                var v = acc + (bias ? bf[c] : 0)
                if flags.contains(.siluGate) { v = v / (1 + exp(-v)) * acc2 }
                if flags.contains(.gelu) { v = 0.5 * v * (1 + tanh(0.7978845608 * (v + 0.044715 * v * v * v))) }
                if flags.contains(.silu) { v = v / (1 + exp(-v)) }
                if flags.contains(.accumulate) { v += y0[r * n + c] }
                want[r * n + c] = v
            }
        }
        assertClose(yb.floats(count: m * n), want, atol: 2e-4 * Float(k).squareRoot() / 30, "gemv m\(m) n\(n) k\(k) flags \(flags.rawValue) bias \(bias)")
    }

    /// The gemm8 kernel assumes this lane-to-element layout of simdgroup_float8x8.
    func testSimdMatrixLayout() throws {
        let m = try f32Buffer((0..<64).map(Float.init), label: "m")
        let out = try ctx.makeBuffer(bytes: 64 * 4, label: "out")
        let cb = ctx.queue.makeCommandBuffer()!
        let ce = cb.makeComputeCommandEncoder()!
        ce.setComputePipelineState(try ctx.pipeline("simd_matrix_layout"))
        ce.setBuffer(m, offset: 0, index: 0)
        ce.setBuffer(out, offset: 0, index: 1)
        ce.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        ce.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let v = out.floats(count: 64)
        for lane in 0..<32 {
            let row = ((lane >> 1) & 3) | ((lane >> 4) << 2)
            let col = ((lane & 1) << 1) | (((lane >> 3) & 1) << 2)
            XCTAssertEqual(Int(v[lane * 2]), row * 8 + col, "lane \(lane) element 0")
            XCTAssertEqual(Int(v[lane * 2 + 1]), row * 8 + col + 1, "lane \(lane) element 1")
        }
    }

    func testGemvShapes() throws {
        try gemvCase(m: 1, n: 1152, k: 896, flags: [], bias: true)
        try gemvCase(m: 2, n: 2, k: 896, flags: [], bias: true)
        try gemvCase(m: 4, n: 896, k: 4864, flags: .accumulate, bias: false)
        try gemvCase(m: 6, n: 4096, k: 1024, flags: .gelu, bias: true)
        try gemvCase(m: 3, n: 64, k: 1024, flags: [], bias: true)
        try gemvCase(m: 21, n: 300, k: 896, flags: [], bias: false)
        try gemvCase(m: 4, n: 3530, k: 896, flags: [], bias: true)
        try gemvCase(m: 1, n: 896, k: 64, flags: [], bias: true)
        try gemvCase(m: 8, n: 2, k: 896, flags: [], bias: true)
    }

    /// Every gemm8 column-tile and K-split variant agrees with the scalar reference, including a K
    /// tail (896 / 8 = 112 = 3 x 32 + 16) that takes the natural-order path per tile, and the two
    /// long-K shapes the engine sends to two tiles (flow ff2, VAE down).
    func testGemm8ColumnTiles() throws {
        defer { Encoder.columnTilesOverride = nil; Encoder.kSplitOverride = nil }
        for nc in [1, 2, 4] {
            for ks in [4, 8] {
                Encoder.columnTilesOverride = nc
                Encoder.kSplitOverride = ks
                try gemvCase(m: 6, n: 1024, k: 1024, flags: .gelu, bias: true)
                try gemvCase(m: 4, n: 896, k: 896, flags: .siluGate, bias: false)
                try gemvCase(m: 3, n: 1152, k: 896, flags: .accumulate, bias: false)
            }
        }
        Encoder.columnTilesOverride = nil
        Encoder.kSplitOverride = nil
        XCTAssertEqual(Encoder.columnTiles(forN: 1024, k: 4096), 2)
        XCTAssertEqual(Encoder.columnTiles(forN: 3530, k: 4096), 1)
        XCTAssertEqual(Encoder.columnTiles(forN: 4096, k: 1024), 1)
        try gemvCase(m: 6, n: 1024, k: 4096, flags: .accumulate, bias: false)
        try gemvCase(m: 4, n: 896, k: 4864, flags: .accumulate, bias: false)
    }

    /// Prefill shapes route through the tiled GEMM (rows >= 16, N % 64 == 0, K % 32 == 0),
    /// including a row count that is not a tile multiple and every epilogue the stack uses.
    func testGemmPrefillShapes() throws {
        XCTAssertTrue(Kernels.usesGemm(rows: 174, n: 1152, k: 896))
        XCTAssertFalse(Kernels.usesGemm(rows: 8, n: 1152, k: 896))
        try gemvCase(m: 174, n: 1152, k: 896, flags: [], bias: true)
        try gemvCase(m: 40, n: 896, k: 4864, flags: .accumulate, bias: false)
        try gemvCase(m: 33, n: 4864, k: 896, flags: .siluGate, bias: false)
        try gemvCase(m: 16, n: 64, k: 32, flags: .gelu, bias: true)
        try gemvCase(m: 65, n: 128, k: 64, flags: .silu, bias: true)
    }

    /// Pre-activations of tens are normal inside the DiT MLP; tanh must not overflow there.
    func testGemvGeluLargeInputs() throws {
        try gemvCase(m: 6, n: 256, k: 1024, flags: .gelu, bias: true, xScale: 8)
        try gemvCase(m: 1, n: 256, k: 896, flags: .silu, bias: false, xScale: 8)
    }

    func testGemvSiluGate() throws {
        try gemvCase(m: 1, n: 4864, k: 896, flags: .siluGate, bias: false)
        try gemvCase(m: 5, n: 256, k: 896, flags: .siluGate, bias: false)
    }

    // MARK: rmsnorm

    func testRMSNorm() throws {
        let rows = 5, k = 896
        var rng = RNG(state: 3)
        let x = rng.floats(rows * k, scale: 3)
        let (w, wf) = try bf16Tensor(rng.floats(k), label: "w")
        let xb = try f32Buffer(x, label: "x")
        let yb = try ctx.makeBuffer(bytes: rows * k * 4, label: "y")
        try run { e in e.rmsnorm(ActView(xb, ld: k), rows: rows, width: k, weight: w, ActView(yb, ld: k)) }
        var want = [Float](repeating: 0, count: rows * k)
        for r in 0..<rows {
            var ss: Float = 0
            for i in 0..<k { ss += x[r * k + i] * x[r * k + i] }
            let s = 1 / (ss / Float(k) + 1e-6).squareRoot()
            for i in 0..<k { want[r * k + i] = x[r * k + i] * s * wf[i] }
        }
        assertClose(yb.floats(count: rows * k), want, atol: 1e-5, "rmsnorm")
    }

    // MARK: rope + cache + attention

    static func neoxRope(_ v: [Float], pos: Int, invFreq: [Float]) -> [Float] {
        var out = v
        for i in 0..<32 {
            let a = Float(pos) * invFreq[i]
            let c = cos(a), s = sin(a)
            out[i] = v[i] * c - v[i + 32] * s
            out[i + 32] = v[i + 32] * c + v[i] * s
        }
        return out
    }

    /// Prefill of `t0` tokens then `m` decode tokens, against a CPU reference over the full sequence.
    func attentionCase(t0: Int, m: Int, window: Int) throws {
        let hq = 14, hkv = 2, ld = (hq + 2 * hkv) * 64, cap = 512
        var rng = RNG(state: UInt64(t0 * 31 + m + window))
        let total = t0 + m
        let qkv = rng.floats((total + 8) * ld)
        let invFreq = (0..<32).map { 1 / powf(1e6, Float(2 * $0) / 64) }
        let invFreqB = try f32Buffer(invFreq, label: "invFreq")
        let kc = try ctx.makeBuffer(bytes: hkv * cap * 64 * 4, label: "kc")
        let vc = try ctx.makeBuffer(bytes: hkv * cap * 64 * 4, label: "vc")
        let cache = KVCacheLayer(k: GPUTensor(buffer: kc, offset: 0, count: 0), v: GPUTensor(buffer: vc, offset: 0, count: 0), capacity: cap)
        let qkvb = try f32Buffer(qkv, label: "qkv")
        let out = try ctx.makeBuffer(bytes: total * hq * 64 * 4, label: "out")
        try run { e in
            for (pos0, rows) in [(0, t0), (t0, m)] where rows > 0 {
                let view = ActView(qkvb, offset: pos0 * ld * 4, ld: ld)
                e.ropeCache(view, rows: rows, heads: hq, kvHeads: hkv, pos0: pos0, cache: cache, invFreq: invFreqB)
                e.attention(view, rows: rows, heads: hq, kvHeads: hkv, pos0: pos0, window: window, cache: cache, ActView(out, offset: pos0 * hq * 64 * 4, ld: hq * 64))
            }
        }
        var want = [Float](repeating: 0, count: total * hq * 64)
        var ks = [[[Float]]](repeating: [], count: hkv)
        for pos in 0..<total {
            for h in 0..<hkv {
                let base = pos * ld + (hq + h) * 64
                ks[h].append(Self.neoxRope(Array(qkv[base..<(base + 64)]), pos: pos, invFreq: invFreq))
            }
        }
        for pos in 0..<total {
            for h in 0..<hq {
                let base = pos * ld + h * 64
                let q = Self.neoxRope(Array(qkv[base..<(base + 64)]), pos: pos, invFreq: invFreq)
                let kvh = h / (hq / hkv)
                let jStart = window > 0 && pos + 1 > window ? pos + 1 - window : 0
                var scores: [Float] = []
                for j in jStart...pos {
                    var s: Float = 0
                    for d in 0..<64 { s += q[d] * ks[kvh][j][d] }
                    scores.append(s / 8)
                }
                let mx = scores.max()!
                let ex = scores.map { exp($0 - mx) }
                let sum = ex.reduce(0, +)
                for (idx, j) in (jStart...pos).enumerated() {
                    let vbase = j * ld + (hq + hkv + kvh) * 64
                    for d in 0..<64 { want[pos * hq * 64 + h * 64 + d] += ex[idx] / sum * qkv[vbase + d] }
                }
            }
        }
        assertClose(out.floats(count: total * hq * 64), want, atol: 2e-5, "attention t0 \(t0) m \(m) window \(window)")
    }

    func testAttentionPrefillAndDecode() throws {
        try attentionCase(t0: 37, m: 1, window: 0)
        try attentionCase(t0: 130, m: 4, window: 0)
    }

    func testAttentionSlidingWindow() throws {
        try attentionCase(t0: 100, m: 4, window: 64)
        try attentionCase(t0: 20, m: 4, window: 64)
    }

    // MARK: DiT attention with interleaved rope

    func testDitAttention() throws {
        let B = 2, S = 3, H = 16, D = H * 64, ld = 3 * D
        var rng = RNG(state: 11)
        let qkv = rng.floats(B * S * ld)
        let invFreq = (0..<32).map { 1 / powf(1e4, Float(2 * $0) / 64) }
        let qkvb = try f32Buffer(qkv, label: "qkv")
        let invFreqB = try f32Buffer(invFreq, label: "invFreq")
        let out = try ctx.makeBuffer(bytes: B * S * D * 4, label: "out")
        try run { e in e.ditAttention(ActView(qkvb, ld: ld), batch: B, seq: S, heads: H, invFreq: invFreqB, ActView(out, ld: D)) }
        func rope(_ v: [Float], pos: Int) -> [Float] {
            var o = v
            for n in 0..<32 {
                let a = Float(pos) * invFreq[n]
                let c = cos(a), s = sin(a)
                o[2 * n] = v[2 * n] * c - v[2 * n + 1] * s
                o[2 * n + 1] = v[2 * n + 1] * c + v[2 * n] * s
            }
            return o
        }
        var want = [Float](repeating: 0, count: B * S * D)
        for b in 0..<B {
            for h in 0..<H {
                for i in 0..<S {
                    let q = rope(Array(qkv[((b * S + i) * ld + h * 64)..<((b * S + i) * ld + h * 64 + 64)]), pos: i)
                    var scores: [Float] = []
                    for j in 0..<S {
                        let kb = (b * S + j) * ld + D + h * 64
                        let k = rope(Array(qkv[kb..<(kb + 64)]), pos: j)
                        scores.append(zip(q, k).map(*).reduce(0, +) / 8)
                    }
                    let mx = scores.max()!
                    let ex = scores.map { exp($0 - mx) }
                    let sum = ex.reduce(0, +)
                    for j in 0..<S {
                        let vb = (b * S + j) * ld + 2 * D + h * 64
                        for d in 0..<64 { want[(b * S + i) * D + h * 64 + d] += ex[j] / sum * qkv[vb + d] }
                    }
                }
            }
        }
        assertClose(out.floats(count: B * S * D), want, atol: 2e-5, "dit attention")
    }

    // MARK: ISTFT

    func testIstftMatchesDirectOverlapAdd() throws {
        let nfft = 3528, hop = 882, bins = nfft / 2 + 1, frames = 8, ring = 8
        var rng = RNG(state: 5)
        var spec = [Float](repeating: 0, count: frames * 2 * bins)
        for f in 0..<frames {
            for k in 0..<bins {
                spec[f * 2 * bins + k] = rng.next() * 2 - 1
                spec[f * 2 * bins + bins + k] = rng.next() * 3
            }
        }
        let window = (0..<nfft).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(nfft))) }
        let cosTable = (0..<nfft).map { Float(cos(2 * Double.pi * Double($0) / Double(nfft))) }
        let specB = try f32Buffer(spec, label: "spec")
        let windowB = try f32Buffer(window, label: "window")
        let cosB = try f32Buffer(cosTable, label: "cos")
        let frameRing = try ctx.makeBuffer(bytes: ring * nfft * 4, label: "frames")
        let total = frames * hop
        let out = try ctx.makeBuffer(bytes: total * 4, label: "out")
        let pad = (nfft - hop) / 2
        try run { e in
            e.istftFrames(spec: ActView(specB, ld: 2 * bins), count: 4, nfft: nfft, cosTable: cosB, window: windowB, frames: frameRing, frame0: 0, ring: ring)
            e.istftEmit(frames: frameRing, window: windowB, nfft: nfft, hop: hop, s0: 0, count: 4 * hop - pad, lastFrame: 3, ring: ring, ActView(out, ld: 1))
            e.istftFrames(spec: ActView(specB, offset: 4 * 2 * bins * 4, ld: 2 * bins), count: 4, nfft: nfft, cosTable: cosB, window: windowB, frames: frameRing, frame0: 4, ring: ring)
            e.istftEmit(frames: frameRing, window: windowB, nfft: nfft, hop: hop, s0: 4 * hop - pad, count: 4 * hop, lastFrame: 7, ring: ring, ActView(out, offset: (4 * hop - pad) * 4, ld: 1))
            e.istftEmit(frames: frameRing, window: windowB, nfft: nfft, hop: hop, s0: 8 * hop - pad, count: pad, lastFrame: 7, ring: ring, ActView(out, offset: (8 * hop - pad) * 4, ld: 1))
        }
        // CPU: irfft per frame in double, window, overlap-add with "same" padding, envelope normalise.
        let full = (frames - 1) * hop + nfft
        var y = [Double](repeating: 0, count: full)
        var env = [Double](repeating: 0, count: full)
        for f in 0..<frames {
            var re = [Double](repeating: 0, count: bins), im = re
            for k in 0..<bins {
                let mag = min(exp(Double(spec[f * 2 * bins + k])), 100)
                let ph = Double(spec[f * 2 * bins + bins + k])
                re[k] = mag * cos(ph); im[k] = mag * sin(ph)
            }
            for t in 0..<nfft {
                var acc = re[0] + (t % 2 == 0 ? re[nfft / 2] : -re[nfft / 2])
                for k in 1..<(nfft / 2) {
                    let a = 2 * Double.pi * Double(k * t % nfft) / Double(nfft)
                    acc += 2 * (re[k] * cos(a) - im[k] * sin(a))
                }
                let w = Double(window[t])
                y[f * hop + t] += acc / Double(nfft) * w
                env[f * hop + t] += w * w
            }
        }
        let want = (0..<total).map { Float(y[$0 + pad] / env[$0 + pad]) }
        let got = out.floats(count: total)
        let scale = want.map(abs).max()!
        assertClose(got, want, atol: 2e-5 * scale, rtol: 1e-4, "istft")
    }

    // MARK: upsample

    func testUpsampleMatchesTorchLinear() throws {
        let hidden = 896, n = 5
        var rng = RNG(state: 9)
        let h = (0..<n).map { _ in rng.floats(hidden) }
        let hb = try h.map { try f32Buffer($0, label: "h") }
        let out = try ctx.makeBuffer(bytes: n * 4 * hidden * 4, label: "frames")
        try run { e in
            for i in 0..<n {
                e.upsample4(prev: i > 0 ? ActView(hb[i - 1], ld: hidden) : nil, cur: ActView(hb[i], ld: hidden),
                            next: i + 1 < n ? ActView(hb[i + 1], ld: hidden) : nil, hidden: hidden, ActView(out, offset: i * 4 * hidden * 4, ld: hidden))
            }
        }
        var want = [Float](repeating: 0, count: n * 4 * hidden)
        for o in 0..<(4 * n) {
            let src = max((Float(o) + 0.5) / 4 - 0.5, 0)
            let i0 = min(Int(src), n - 1), i1 = min(i0 + 1, n - 1)
            let lambda = src - Float(i0)
            for c in 0..<hidden { want[o * hidden + c] = (1 - lambda) * h[i0][c] + lambda * h[i1][c] }
        }
        assertClose(out.floats(count: n * 4 * hidden), want, atol: 1e-6, "upsample")
    }
}

