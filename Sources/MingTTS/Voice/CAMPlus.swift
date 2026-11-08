// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Accelerate
import Foundation

/// CAM++, the 192-dim speaker net in `campplus.onnx`, run on the CPU from the ONNX initializers.
/// A port of a NumPy reimplementation of CAM++ that was checked against onnxruntime.
struct CAMPlus {
    struct Conv {
        let weight: [Float]
        let outChannels: Int
        let kernel: (h: Int, w: Int)
        let bias: [Float]?
    }

    struct Affine {
        let scale: [Float]
        let shift: [Float]
    }

    struct ResBlock {
        let conv1: Conv, conv2: Conv, shortcut: Conv?
        let stride: Int
    }

    struct DenseLayer {
        let norm: Affine
        let linear1: Conv, local: Conv, context1: Conv, context2: Conv
        let dilation: Int
    }

    struct DenseBlock {
        let layers: [DenseLayer]
        let transitNorm: Affine
        let transit: Conv
    }

    static let segment = 100
    static let blockTable: [(name: String, layers: Int, dilation: Int)] = [("block1", 12, 1), ("block2", 24, 2), ("block3", 16, 2)]

    let headConv1: Conv, headConv2: Conv
    let resBlocks: [ResBlock]
    let tdnn: Conv
    let blocks: [DenseBlock]
    let dense: Conv
    let denseNorm: Affine

    init(onnx: URL) throws {
        let g = try ONNXWeights(url: onnx)
        func conv(_ name: String) throws -> Conv {
            let (w, b) = try g.conv(name)
            return Conv(weight: w.values, outChannels: w.dims[0], kernel: (w.dims.count == 4 ? w.dims[2] : 1, w.dims.last!), bias: b)
        }
        func affine(_ name: String) throws -> Affine {
            let (s, t) = try g.batchNorm(name)
            return Affine(scale: s, shift: t)
        }
        headConv1 = try conv("/head/conv1/Conv")
        resBlocks = try ["layer1", "layer2"].flatMap { layer in
            try (0..<2).map { block in
                let p = "/head/\(layer)/\(layer).\(block)"
                return ResBlock(conv1: try conv("\(p)/conv1/Conv"), conv2: try conv("\(p)/conv2/Conv"),
                                shortcut: block == 0 ? try conv("\(p)/shortcut/shortcut.0/Conv") : nil, stride: block == 0 ? 2 : 1)
            }
        }
        headConv2 = try conv("/head/conv2/Conv")
        tdnn = try conv("/xvector/tdnn/linear/Conv")
        blocks = try Self.blockTable.map { name, count, dilation in
            let layers = try (1...count).map { i in
                let p = "/xvector/\(name)/tdnnd\(i)"
                return DenseLayer(norm: try affine("\(p)/nonlinear1/batchnorm/BatchNormalization"),
                                  linear1: try conv("\(p)/linear1/Conv"), local: try conv("\(p)/cam_layer/linear_local/Conv"),
                                  context1: try conv("\(p)/cam_layer/linear1/Conv"), context2: try conv("\(p)/cam_layer/linear2/Conv"),
                                  dilation: dilation)
            }
            let t = "/xvector/\(name.replacingOccurrences(of: "block", with: "transit"))"
            return DenseBlock(layers: layers, transitNorm: try affine("\(t)/nonlinear/batchnorm/BatchNormalization"), transit: try conv("\(t)/linear/Conv"))
        }
        dense = try conv("/xvector/dense/linear/Conv")
        denseNorm = try affine("/xvector/dense/nonlinear/batchnorm/BatchNormalization")
    }

    /// `fbank` row-major `[frames, 80]`, already mean-normalised over time; returns 192 floats.
    func embed(fbank: [Float], frames: Int) -> [Float] {
        let mels = fbank.count / frames
        var x = [Float](repeating: 0, count: fbank.count)
        vDSP_mtrans(fbank, 1, &x, 1, vDSP_Length(mels), vDSP_Length(frames))
        var freq = mels
        (x, freq) = conv2d(x, freq: freq, time: frames, headConv1, stride: 1, pad: 1)
        relu(&x)
        for block in resBlocks {
            var (h, f) = conv2d(x, freq: freq, time: frames, block.conv1, stride: block.stride, pad: 1)
            relu(&h)
            (h, f) = conv2d(h, freq: f, time: frames, block.conv2, stride: 1, pad: 1)
            let shortcut = block.shortcut.map { conv2d(x, freq: freq, time: frames, $0, stride: block.stride, pad: 0).out } ?? x
            vDSP.add(h, shortcut, result: &h)
            relu(&h)
            (x, freq) = (h, f)
        }
        (x, freq) = conv2d(x, freq: freq, time: frames, headConv2, stride: 2, pad: 1)
        relu(&x)
        var t: Int
        (x, t) = conv1d(x, time: frames, tdnn, stride: 2, pad: 2)
        relu(&x)
        for block in blocks {
            for layer in block.layers {
                var h = x
                normalize(&h, layer.norm, time: t)
                relu(&h)
                h = conv1d(h, time: t, layer.linear1).out
                relu(&h)
                let local = conv1d(h, time: t, layer.local, dilation: layer.dilation, pad: layer.dilation).out
                var m = conv1d(contextMeans(h, time: t), time: t, layer.context1).out
                relu(&m)
                m = conv1d(m, time: t, layer.context2).out
                for i in m.indices { m[i] = local[i] / (1 + exp(-m[i])) }
                x += m
            }
            normalize(&x, block.transitNorm, time: t)
            relu(&x)
            x = conv1d(x, time: t, block.transit).out
        }
        relu(&x)
        let channels = x.count / t
        var stats = [Float](repeating: 0, count: 2 * channels)
        for c in 0..<channels {
            let row = x[(c * t)..<((c + 1) * t)]
            let mean = vDSP.mean(row)
            stats[c] = mean
            stats[channels + c] = (row.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(t - 1)).squareRoot()
        }
        var e = conv1d(stats, time: 1, dense).out
        normalize(&e, denseNorm, time: 1)
        return e
    }

    /// Mean over all frames plus the mean of each `segment`-frame chunk (the last over its remainder).
    private func contextMeans(_ h: [Float], time t: Int) -> [Float] {
        var out = [Float](repeating: 0, count: h.count)
        for c in 0..<(h.count / t) {
            let row = c * t
            let global = vDSP.mean(h[row..<(row + t)])
            for s in stride(from: 0, to: t, by: Self.segment) {
                let end = min(s + Self.segment, t)
                let local = vDSP.mean(h[(row + s)..<(row + end)])
                for i in s..<end { out[row + i] = global + local }
            }
        }
        return out
    }
}

private func relu(_ x: inout [Float]) {
    vDSP.threshold(x, to: 0, with: .clampToThreshold, result: &x)
}

/// `x` is `[C, T]`; applies a per-channel affine in place.
private func normalize(_ x: inout [Float], _ a: CAMPlus.Affine, time t: Int) {
    for c in 0..<a.scale.count {
        for i in (c * t)..<((c + 1) * t) { x[i] = x[i] * a.scale[c] + a.shift[c] }
    }
}

/// `x` is `[C, F, T]`; zero padding of `pad` on both axes, stride along frequency only.
private func conv2d(_ x: [Float], freq: Int, time t: Int, _ p: CAMPlus.Conv, stride: Int, pad: Int) -> (out: [Float], freq: Int) {
    let channels = x.count / (freq * t)
    let (kh, kw) = p.kernel
    let outFreq = (freq + 2 * pad - kh) / stride + 1
    var cols = [Float](repeating: 0, count: channels * kh * kw * outFreq * t)
    var r = 0
    for c in 0..<channels {
        for i in 0..<kh {
            for j in 0..<kw {
                for fo in 0..<outFreq {
                    let fi = fo * stride + i - pad
                    if fi >= 0, fi < freq {
                        let src = (c * freq + fi) * t, dst = (r * outFreq + fo) * t
                        for to in max(0, pad - j)..<min(t, t + pad - j) { cols[dst + to] = x[src + to + j - pad] }
                    }
                }
                r += 1
            }
        }
    }
    return (gemm(p, cols, columns: outFreq * t), outFreq)
}

/// `x` is `[C, T]`; returns `[O, T_out]`.
private func conv1d(_ x: [Float], time t: Int, _ p: CAMPlus.Conv, stride: Int = 1, dilation: Int = 1, pad: Int = 0) -> (out: [Float], time: Int) {
    let k = p.kernel.w
    if k == 1, stride == 1 { return (gemm(p, x, columns: t), t) }
    let channels = x.count / t
    let outTime = (t + 2 * pad - dilation * (k - 1) - 1) / stride + 1
    var cols = [Float](repeating: 0, count: channels * k * outTime)
    for c in 0..<channels {
        for i in 0..<k {
            let dst = (c * k + i) * outTime
            for to in 0..<outTime {
                let ti = to * stride + i * dilation - pad
                if ti >= 0, ti < t { cols[dst + to] = x[c * t + ti] }
            }
        }
    }
    return (gemm(p, cols, columns: outTime), outTime)
}

/// `weight [O, K] @ cols [K, columns]` plus the bias per output row.
private func gemm(_ p: CAMPlus.Conv, _ cols: [Float], columns n: Int) -> [Float] {
    let o = p.outChannels, k = p.weight.count / o
    var out = [Float](repeating: 0, count: o * n)
    if let bias = p.bias {
        for r in 0..<o { for i in 0..<n { out[r * n + i] = bias[r] } }
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(o), Int32(n), Int32(k),
                1, p.weight, Int32(k), cols, Int32(n), p.bias == nil ? 0 : 1, &out, Int32(n))
    return out
}
