// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Isolated timing of the decode-time GEMV shapes: `repeats` back-to-back serial dispatches per shape,
/// each reading a fresh slice of a 1 GiB weight pool so no cache level can serve the weights twice,
/// under each variant of MING_BENCH_VARIANTS, interleaved. An empty dispatch gives the fixed-cost floor.
public enum GemvBench {
    struct Shape {
        let label: String
        let m, n, k: Int
        let flags: GemvFlags
        let gated: Bool
        var bytes: Int { n * k * 2 * (gated ? 2 : 1) }
    }

    static let shapes: [Shape] = [
        Shape(label: "llm qkv", m: 1, n: 1152, k: 896, flags: [], gated: false),
        Shape(label: "llm o", m: 1, n: 896, k: 896, flags: .accumulate, gated: false),
        Shape(label: "llm gate|up", m: 1, n: 4864, k: 896, flags: .siluGate, gated: true),
        Shape(label: "llm down", m: 1, n: 896, k: 4864, flags: .accumulate, gated: false),
        Shape(label: "agg qkv", m: 2, n: 3072, k: 1024, flags: [], gated: false),
        Shape(label: "agg out", m: 2, n: 1024, k: 1024, flags: .accumulate, gated: false),
        Shape(label: "agg ff1", m: 2, n: 4096, k: 1024, flags: .gelu, gated: false),
        Shape(label: "agg ff2", m: 2, n: 1024, k: 4096, flags: .accumulate, gated: false),
        Shape(label: "vae qkv", m: 4, n: 1152, k: 896, flags: [], gated: false),
        Shape(label: "vae o", m: 4, n: 896, k: 896, flags: .accumulate, gated: false),
        Shape(label: "vae gate|up", m: 4, n: 4864, k: 896, flags: .siluGate, gated: true),
        Shape(label: "vae down", m: 4, n: 896, k: 4864, flags: .accumulate, gated: false),
        Shape(label: "vae head", m: 4, n: 3530, k: 896, flags: [], gated: false),
        Shape(label: "flow qkv", m: 6, n: 3072, k: 1024, flags: [], gated: false),
        Shape(label: "flow out", m: 6, n: 1024, k: 1024, flags: .accumulate, gated: false),
        Shape(label: "flow ff1", m: 6, n: 4096, k: 1024, flags: .gelu, gated: false),
        Shape(label: "flow ff2", m: 6, n: 1024, k: 4096, flags: .accumulate, gated: false),
    ]

    public static func layout() throws {
        let ctx = try MetalContext()
        let m = try ctx.makeBuffer(floats: (0..<64).map(Float.init), label: "m")
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
            let a = Int(v[lane * 2]), b = Int(v[lane * 2 + 1])
            print("lane \(lane): (\(a / 8),\(a % 8)) (\(b / 8),\(b % 8))")
        }
    }

    /// One configuration of the dispatch path, from tokens joined by "+": "engine" (the engine's own
    /// kernel choice), "ks4" / "ks8" / "ks16" (gemm8 K split), "scalar" (scalar kernel up to 8 rows),
    /// "nc2" / "nc4" (gemm8 column tiles per simdgroup), "gemm8" (gemm8 even at 1-2 rows), "pad8" (X row
    /// stride K + 8, to separate cache-set conflicts between rows a power of two apart from the load cost).
    struct Variant {
        let label: String
        var kSplit: Int?
        var columnTiles: Int?
        var scalarRowsMax = 2
        var xPad = 0

        init(_ spec: String) throws {
            label = spec
            for tok in spec.split(separator: "+").map(String.init) {
                if tok == "engine" { continue }
                if tok == "scalar" { scalarRowsMax = 8; continue }
                if tok == "gemm8" { scalarRowsMax = 0; continue }
                if tok.hasPrefix("ks"), let v = Int(tok.dropFirst(2)) { kSplit = v; continue }
                if tok.hasPrefix("nc"), let v = Int(tok.dropFirst(2)) { columnTiles = v; continue }
                if tok.hasPrefix("pad"), let v = Int(tok.dropFirst(3)) { xPad = v; continue }
                throw MingTTSError.gpuFailure("unknown gemv-bench variant token \(tok)")
            }
        }

        func apply() {
            Encoder.kSplitOverride = kSplit
            Encoder.columnTilesOverride = columnTiles
            Encoder.scalarRowsMax = scalarRowsMax
        }
    }

    /// Serial dispatches (one compute encoder, the engine's situation) of every shape under each
    /// variant in MING_BENCH_VARIANTS (comma separated, default "engine"), interleaved per shape so
    /// the variants see the same machine noise: `rounds` command buffers of `repeats` dispatches
    /// each, alternating variant order per round, median per variant.
    public static func run(repeats: Int = 200, rounds: Int = 5) throws {
        let ctx = try MetalContext()
        let kernels = try Kernels(ctx: ctx)
        let poolBytes = 1 << 30
        let pool = try ctx.makeBuffer(bytes: poolBytes, label: "bench.pool")
        fillBF16(pool)
        let x = try ctx.makeBuffer(bytes: 16 * 4864 * 4, label: "bench.x")
        fillF32(x)
        let y = try ctx.makeBuffer(bytes: 16 * 4864 * 4, label: "bench.y")
        let bias = GPUTensor(buffer: pool, offset: 0, count: 4864)
        let specs = (ProcessInfo.processInfo.environment["MING_BENCH_VARIANTS"] ?? "engine").split(separator: ",").map(String.init)
        let variants = try specs.map(Variant.init)

        func time(_ body: (Encoder) -> Void) throws -> Double {
            let cb = ctx.queue.makeCommandBuffer()!
            let ce = cb.makeComputeCommandEncoder()!
            body(Encoder(k: kernels, ce: ce))
            ce.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw MingTTSError.gpuFailure("\(e)") }
            return (cb.gpuEndTime - cb.gpuStartTime) / Double(repeats) * 1e6
        }

        var floors: [Double] = []
        for _ in 0..<rounds {
            floors.append(try time { e in
                for _ in 0..<repeats { e.copy(ActView(x, ld: 1), count: 1, to: ActView(y, ld: 1)) }
            })
        }
        print(String(format: "%d dispatches per shape, median of %d interleaved command buffers, us per dispatch (GB/s of weights); empty dispatch floor %.1f us", repeats, rounds, floors.sorted()[rounds / 2]))
        print("shape            M     N     K    MB  " + variants.map { $0.label.padding(toLength: 18, withPad: " ", startingAt: 0) }.joined())
        for s in shapes {
            var offset = 0
            func next() -> Int {
                if offset + s.bytes > poolBytes { offset = 0 }
                defer { offset += s.bytes }
                return offset
            }
            func encodeAll(_ v: Variant) -> (Encoder) -> Void {
                return { e in
                    offset = 0
                    v.apply()
                    for _ in 0..<repeats {
                        let o = next()
                        let lin = Linear(w: GPUTensor(buffer: pool, offset: o, count: s.n * s.k), bias: s.flags.contains(.accumulate) ? nil : bias, n: s.n, k: s.k)
                        let gate = s.gated ? GPUTensor(buffer: pool, offset: o + s.n * s.k * 2, count: s.n * s.k) : nil
                        e.linear(ActView(x, ld: s.k + v.xPad), rows: s.m, lin, ActView(y, ld: s.n), flags: s.flags, gate: gate)
                    }
                    Encoder.kSplitOverride = nil
                    Encoder.columnTilesOverride = nil
                    Encoder.scalarRowsMax = 2
                }
            }
            var samples = [[Double]](repeating: [], count: variants.count)
            for r in 0..<rounds {
                let order = r % 2 == 0 ? Array(variants.indices) : Array(variants.indices.reversed())
                for i in order { samples[i].append(try time(encodeAll(variants[i]))) }
            }
            let mb = Double(s.bytes) / 1e6
            let cells = samples.map { v -> String in
                let med = v.sorted()[rounds / 2]
                return String(format: "%9.1f (%4.0f)   ", med, mb * 1e3 / med)
            }
            print(String(format: "%-14@ %4d %5d %5d %5.1f  ", s.label as NSString, s.m, s.n, s.k, mb) + cells.joined())
        }
    }

    /// bf16 values in [-0.5, -0.125] and [0.125, 0.5): a valid exponent range, both signs.
    static func fillBF16(_ b: MTLBuffer) {
        let n = b.length / 2
        let p = b.contents().bindMemory(to: UInt16.self, capacity: n)
        var s: UInt64 = 0x9E3779B97F4A7C15
        for i in 0..<n {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            p[i] = (UInt16(truncatingIfNeeded: s >> 48) & 0x81FF) | 0x3E00
        }
    }

    static func fillF32(_ b: MTLBuffer) {
        let n = b.length / 4
        let p = b.contents().bindMemory(to: Float.self, capacity: n)
        var s: UInt64 = 12345
        for i in 0..<n {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            p[i] = Float(s >> 40) / Float(1 << 24) * 2 - 1
        }
    }
}
