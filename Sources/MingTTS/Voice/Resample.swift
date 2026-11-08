// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Accelerate
import Foundation

/// torchaudio 2.6 `Resample(orig, new)` with its defaults: windowed-sinc (Hann), lowpass filter
/// width 6, rolloff 0.99. The kernel is built in float64 and rounded to float32, as torchaudio does.
func resample(_ samples: [Float], from orig: Int, to new: Int) -> [Float] {
    if orig == new { return samples }
    let divisor = gcd(orig, new)
    let o = orig / divisor, n = new / divisor
    let (kernels, width) = sincKernels(orig: o, new: n)
    let taps = 2 * width + o
    let padded = [Float](repeating: 0, count: width) + samples + [Float](repeating: 0, count: width + o)
    // torchaudio rounds the exact ratio to float32 before the ceil, which can drop a sample.
    let target = Int(Float(Double(n * samples.count) / Double(o)).rounded(.up))
    var out = [Float](repeating: 0, count: target)
    padded.withUnsafeBufferPointer { x in
        kernels.withUnsafeBufferPointer { k in
            for i in 0..<target {
                vDSP_dotpr(x.baseAddress! + (i / n) * o, 1, k.baseAddress! + (i % n) * taps, 1, &out[i], vDSP_Length(taps))
            }
        }
    }
    return out
}

/// `new` phase kernels of `2 * width + orig` taps each, row-major.
private func sincKernels(orig: Int, new: Int) -> (kernels: [Float], width: Int) {
    let filterWidth = 6.0, rolloff = 0.99
    let base = Double(min(orig, new)) * rolloff
    let width = Int((filterWidth * Double(orig) / base).rounded(.up))
    let taps = 2 * width + orig
    var kernels = [Float](repeating: 0, count: new * taps)
    for j in 0..<new {
        // torchaudio's phase offset is an integer arange divided in float32, then promoted to float64.
        let phase = Double(Float(-j) / Float(new))
        for k in 0..<taps {
            var t = (phase + Double(k - width) / Double(orig)) * base
            t = min(max(t, -filterWidth), filterWidth)
            let c = cos(t * .pi / filterWidth / 2)
            t *= .pi
            let sinc = t == 0 ? 1 : sin(t) / t
            kernels[j * taps + k] = Float(sinc * (c * c * (base / Double(orig))))
        }
    }
    return (kernels, width)
}

private func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
