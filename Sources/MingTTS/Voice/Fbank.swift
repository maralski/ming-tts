// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Accelerate
import Foundation

/// torchaudio `kaldi.fbank(wave, num_mel_bins=80, dither=0, sample_frequency=16000)` with the mean
/// over time subtracted, as the speaker extractor feeds CAM++. Returns row-major `[frames, 80]`.
func kaldiFbank(_ wave16k: [Float]) -> (features: [Float], frames: Int) {
    let size = 400, shift = 160, fftSize = 512, bins = 256, mels = 80
    guard wave16k.count >= size else { return ([], 0) }
    let frames = 1 + (wave16k.count - size) / shift
    let window = poveyWindow(size)
    let dft = vDSP.DFT(count: fftSize, direction: .forward, transformType: .complexComplex, ofType: Float.self)!
    var power = [Float](repeating: 0, count: frames * bins)
    var frame = [Float](repeating: 0, count: fftSize)
    let zeros = [Float](repeating: 0, count: fftSize)
    var re = [Float](repeating: 0, count: fftSize), im = [Float](repeating: 0, count: fftSize)
    for f in 0..<frames {
        let raw = wave16k[(f * shift)..<(f * shift + size)]
        let mean = vDSP.mean(raw)
        var previous = raw.first! - mean
        for (j, x) in raw.enumerated() {
            let centred = x - mean
            frame[j] = (centred - 0.97 * previous) * window[j]
            previous = centred
        }
        dft.transform(inputReal: frame, inputImaginary: zeros, outputReal: &re, outputImaginary: &im)
        for k in 0..<bins { power[f * bins + k] = re[k] * re[k] + im[k] * im[k] }
    }
    var energies = [Float](repeating: 0, count: frames * mels)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(frames), Int32(mels), Int32(bins),
                1, power, Int32(bins), melBanks(mels: mels, bins: bins), Int32(bins), 0, &energies, Int32(mels))
    for i in energies.indices { energies[i] = log(max(energies[i], Float.ulpOfOne)) }
    for m in 0..<mels {
        var sum: Float = 0
        for f in 0..<frames { sum += energies[f * mels + m] }
        let mean = sum / Float(frames)
        for f in 0..<frames { energies[f * mels + m] -= mean }
    }
    return (energies, frames)
}

/// Kaldi's povey window: a symmetric Hann raised to 0.85.
private func poveyWindow(_ size: Int) -> [Float] {
    (0..<size).map { pow(0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(size - 1)), 0.85) }
}

/// Triangular filters on the Kaldi mel scale from 20 Hz to Nyquist, row-major `[mels, bins]`.
private func melBanks(mels: Int, bins: Int) -> [Float] {
    func mel(_ hz: Double) -> Double { 1127 * log(1 + hz / 700) }
    let sampleRate = 16000.0, lowHz = 20.0
    let low = mel(lowHz), high = mel(sampleRate / 2)
    let delta = (high - low) / Double(mels + 1)
    let binWidth = sampleRate / Double(2 * bins)
    var banks = [Float](repeating: 0, count: mels * bins)
    for m in 0..<mels {
        let left = Float(low + Double(m) * delta), centre = Float(low + Double(m + 1) * delta), right = Float(low + Double(m + 2) * delta)
        for k in 0..<bins {
            let f = Float(1127) * log(1 + Float(binWidth) * Float(k) / 700)
            banks[m * bins + k] = max(0, min((f - left) / (centre - left), (right - f) / (right - centre)))
        }
    }
    return banks
}
