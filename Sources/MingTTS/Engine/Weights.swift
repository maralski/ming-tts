// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Copies the bf16 tensors one module needs into a single GPU buffer, 256-byte aligned.
/// The builder runs twice: once to size the buffer, once to copy.
final class WeightArena {
    private let st: SafeTensors
    private var planning = true
    private var cursor = 0
    private(set) var buffer: MTLBuffer?

    private init(st: SafeTensors) { self.st = st }

    static func load<T>(_ st: SafeTensors, ctx: MetalContext, label: String, _ build: (WeightArena) throws -> T) throws -> T {
        let arena = WeightArena(st: st)
        _ = try build(arena)
        arena.buffer = try ctx.makeBuffer(bytes: arena.cursor, label: label)
        arena.planning = false
        arena.cursor = 0
        return try build(arena)
    }

    private func place(bytes: Int) -> Int {
        let offset = cursor
        cursor += (bytes + 255) & ~255
        return offset
    }

    /// A bf16 tensor, or several concatenated along their first axis (q, k, v rows for instance).
    func tensor(_ names: String...) throws -> GPUTensor {
        var total = 0
        for name in names {
            let e = try st.entry(name)
            guard e.dtype == "BF16" else { throw MingTTSError.badCheckpoint("\(name) is \(e.dtype), expected BF16") }
            total += e.byteRange.count
        }
        let offset = place(bytes: total)
        if let buffer, !planning {
            var dst = buffer.contents() + offset
            for name in names {
                try st.withBytes(name) { raw, _ in
                    memcpy(dst, raw.baseAddress!, raw.count)
                    dst += raw.count
                }
            }
        }
        return GPUTensor(buffer: buffer ?? placeholder, offset: offset, count: total / 2)
    }

    func linear(_ prefix: String, n: Int, k: Int, bias: Bool) throws -> Linear {
        let w = try tensor(prefix + ".weight")
        guard w.count == n * k else { throw MingTTSError.badCheckpoint("\(prefix).weight has \(w.count) elements, expected \(n)x\(k)") }
        return Linear(w: w, bias: bias ? try tensor(prefix + ".bias") : nil, n: n, k: k)
    }

    /// A bf16 [n, storedK] weight laid out as [n, k] with zero columns past storedK, for a shape
    /// the kernels cannot stride (they read K eight values at a time).
    func paddedLinear(_ prefix: String, n: Int, k: Int, storedK: Int) throws -> Linear {
        let e = try st.entry(prefix + ".weight")
        guard e.dtype == "BF16", e.shape == [n, storedK] else { throw MingTTSError.badCheckpoint("\(prefix).weight is \(e.dtype) \(e.shape), expected BF16 [\(n), \(storedK)]") }
        let offset = place(bytes: n * k * 2)
        if let buffer, !planning {
            let dst = buffer.contents() + offset
            memset(dst, 0, n * k * 2)
            try st.withBytes(prefix + ".weight") { raw, _ in
                for row in 0..<n { memcpy(dst + row * k * 2, raw.baseAddress! + row * storedK * 2, storedK * 2) }
            }
        }
        return Linear(w: GPUTensor(buffer: buffer ?? placeholder, offset: offset, count: n * k), bias: nil, n: n, k: k)
    }

    /// Stand-in during the sizing pass; never bound to an encoder.
    private var placeholder: MTLBuffer {
        struct Holder { nonisolated(unsafe) static let buffer = MTLCreateSystemDefaultDevice()!.makeBuffer(length: 16)! }
        return Holder.buffer
    }
}

struct QwenConfig {
    var hidden = 896
    var heads = 14
    var kvHeads = 2
    var intermediate = 4864
    var layers = 24
    var ropeTheta: Float = 1e6
    var eps: Float = 1e-6
    /// 0 means full causal attention.
    var window = 0
    var qkvWidth: Int { (heads + 2 * kvHeads) * 64 }
}

struct QwenLayerWeights {
    let norm1: GPUTensor
    let qkv: Linear
    let o: Linear
    let norm2: GPUTensor
    let gate: Linear
    let up: Linear
    let down: Linear
}

/// The Qwen2 decoder stack shared by the LLM and the VAE decoder.
struct QwenWeights {
    let config: QwenConfig
    let layers: [QwenLayerWeights]
    let finalNorm: GPUTensor

    init(config: QwenConfig, layers: [QwenLayerWeights], finalNorm: GPUTensor) {
        self.config = config
        self.layers = layers
        self.finalNorm = finalNorm
    }

    init(_ a: WeightArena, prefix: String, config: QwenConfig) throws {
        self.config = config
        let h = config.hidden, kv = config.kvHeads * 64, inter = config.intermediate
        layers = try (0..<config.layers).map { i in
            let p = "\(prefix).layers.\(i)"
            let qkvW = try a.tensor("\(p).self_attn.q_proj.weight", "\(p).self_attn.k_proj.weight", "\(p).self_attn.v_proj.weight")
            let qkvB = try a.tensor("\(p).self_attn.q_proj.bias", "\(p).self_attn.k_proj.bias", "\(p).self_attn.v_proj.bias")
            return QwenLayerWeights(
                norm1: try a.tensor("\(p).input_layernorm.weight"),
                qkv: Linear(w: qkvW, bias: qkvB, n: h + 2 * kv, k: h),
                o: try a.linear("\(p).self_attn.o_proj", n: h, k: h, bias: false),
                norm2: try a.tensor("\(p).post_attention_layernorm.weight"),
                gate: try a.linear("\(p).mlp.gate_proj", n: inter, k: h, bias: false),
                up: try a.linear("\(p).mlp.up_proj", n: inter, k: h, bias: false),
                down: try a.linear("\(p).mlp.down_proj", n: h, k: inter, bias: false))
        }
        finalNorm = try a.tensor("\(prefix).norm.weight")
    }
}

struct DiTBlockWeights {
    let norm1: GPUTensor
    let qkv: Linear
    let out: Linear
    let norm2: GPUTensor
    let ff1: Linear
    let ff2: Linear
}

/// The DiT block stack shared by the flow head and the aggregator (hidden 1024, 16 heads).
struct DiTWeights {
    static let hidden = 1024
    static let heads = 16
    let blocks: [DiTBlockWeights]
    let finalNorm: GPUTensor
    let finalLinear: Linear
    let invFreq: [Float]

    init(_ a: WeightArena, st: SafeTensors, prefix: String, depth: Int, outDim: Int) throws {
        let h = Self.hidden
        blocks = try (0..<depth).map { i in
            let p = "\(prefix).blocks.\(i)"
            let qkvW = try a.tensor("\(p).attn.to_q.weight", "\(p).attn.to_k.weight", "\(p).attn.to_v.weight")
            let qkvB = try a.tensor("\(p).attn.to_q.bias", "\(p).attn.to_k.bias", "\(p).attn.to_v.bias")
            return DiTBlockWeights(
                norm1: try a.tensor("\(p).norm1.weight"),
                qkv: Linear(w: qkvW, bias: qkvB, n: 3 * h, k: h),
                out: try a.linear("\(p).attn.to_out.0", n: h, k: h, bias: true),
                norm2: try a.tensor("\(p).norm2.weight"),
                ff1: try a.linear("\(p).mlp.ff.0.0", n: 4 * h, k: h, bias: true),
                ff2: try a.linear("\(p).mlp.ff.2", n: h, k: 4 * h, bias: true))
        }
        finalNorm = try a.tensor("\(prefix).final_layer.norm_final.weight")
        finalLinear = try a.linear("\(prefix).final_layer.linear", n: outDim, k: h, bias: true)
        invFreq = try st.floats("\(prefix).rotary_embed.inv_freq")
    }
}

struct FlowHeadWeights {
    let xEmbedder: Linear
    let condEmbedder: Linear
    let time1: Linear
    let time2: Linear
    let dit: DiTWeights

    init(_ a: WeightArena, st: SafeTensors) throws {
        let p = "flowloss.cfm.model"
        xEmbedder = try a.linear("\(p).x_embedder", n: 1024, k: 64, bias: true)
        condEmbedder = try a.linear("\(p).c_embedder.cond_embedder", n: 1024, k: 896, bias: true)
        time1 = try a.linear("\(p).t_embedder.time_mlp.0", n: 1024, k: 256, bias: true)
        time2 = try a.linear("\(p).t_embedder.time_mlp.2", n: 1024, k: 1024, bias: true)
        dit = try DiTWeights(a, st: st, prefix: p, depth: 8, outDim: 64)
    }
}

struct AggregatorWeights {
    let xEmbedder: Linear
    let cls: [Float]
    let dit: DiTWeights

    init(_ a: WeightArena, st: SafeTensors) throws {
        let p = "linear_proj_audio"
        xEmbedder = try a.linear("\(p).x_embedder", n: 1024, k: 64, bias: true)
        cls = try st.floats("\(p).word_embedder.weight")
        dit = try DiTWeights(a, st: st, prefix: p, depth: 8, outDim: 896)
    }
}

struct VAEDecoderWeights {
    static let nfft = 3528
    static let hop = 882
    let fc1: Linear
    let stack: QwenWeights
    let head: Linear
    let window: [Float]

    init(_ a: WeightArena, st: SafeTensors) throws {
        let p = "audio.decoder"
        fc1 = try a.linear("\(p).fc1", n: 896, k: 64, bias: true)
        var cfg = QwenConfig()
        // The reference attends 65 keys per query (manifest vae_attention_window_keys), not 64.
        cfg.window = 65
        stack = try QwenWeights(a, prefix: "\(p).decoder", config: cfg)
        head = try a.linear("\(p).head.out", n: Self.nfft + 2, k: 896, bias: true)
        window = try st.floats("\(p).head.istft.window")
    }
}

/// The audio VAE encoder: frame projections, the 24-layer Qwen stack, the 4-layer patch aggregator
/// with its cls row, and the posterior head. `audio.encoder.norm` exists in the checkpoint but the
/// reference forward never calls it.
struct VAEEncoderWeights {
    static let frame = 882
    static let patch = 4
    static let hidden = 896
    /// Frame samples padded to this many columns so the GEMM kernels can stride them.
    static let paddedFrame = 896
    let fc1: Linear
    let fc2: Linear
    let stack: QwenWeights
    let aggregator: QwenWeights
    let cls: [Float]
    let fc3: Linear

    init(_ a: WeightArena, st: SafeTensors) throws {
        let p = "audio.encoder"
        fc1 = try a.paddedLinear("\(p).fc1", n: Self.hidden, k: Self.paddedFrame, storedK: Self.frame)
        fc2 = try a.linear("\(p).fc2", n: Self.hidden, k: Self.hidden, bias: true)
        var cfg = QwenConfig()
        // 65 keys per query, as the decoder (manifest vae_attention_window_keys); both stacks share one config.
        cfg.window = 65
        stack = try QwenWeights(a, prefix: "\(p).encoder", config: cfg)
        cfg.layers = 4
        aggregator = try QwenWeights(a, prefix: "\(p).aggregator", config: cfg)
        cls = try st.floats("\(p).cls_embed")
        fc3 = try a.linear("\(p).fc3", n: 2 * MingEngine.latentDim, k: Self.hidden, bias: true)
    }
}

struct LLMWeights {
    let stack: QwenWeights
    let stopHead: Linear

    init(_ a: WeightArena) throws {
        stack = try QwenWeights(a, prefix: "model.model", config: QwenConfig())
        stopHead = try a.linear("stop_head", n: 2, k: 896, bias: true)
    }
}
