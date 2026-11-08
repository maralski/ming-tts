// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// `frontend.audio_processing.trim_trailing_silence`: scan 0.1 s frames backwards for the last
/// one whose mean absolute sample is above 1e-3 and keep 0.3 s beyond it. Silent throughout, or
/// shorter than one frame, keeps the fixed tail.
func trimTrailingSilence(_ speech: [Float], sampleRate: Int, threshold: Float = 1e-3, tailSeconds: Double = 0.3) -> [Float] {
    if speech.isEmpty { return speech }
    let frame = Int(Double(sampleRate) * 0.1), tail = Int(Double(sampleRate) * tailSeconds)
    if frame <= 0 || speech.count < frame { return Array(speech.prefix(tail)) }
    let frames = (speech.count - frame) / frame + 1
    var index = frames - 1
    while index >= 0 {
        let start = index * frame
        let score = speech[start..<(start + frame)].reduce(0) { $0 + abs($1) } / Float(frame)
        if score > threshold { break }
        index -= 1
    }
    if index < 0 { return Array(speech.prefix(tail)) }
    return Array(speech.prefix(min(frames * frame, index * frame + frame + tail)))
}

/// `speech` from 50 ms before its first to 50 ms after its last 10 ms frame whose RMS is above 1% of the peak (-40 dB).
/// Silent throughout keeps nothing.
func trimSilence(_ speech: [Float], sampleRate: Int, padSeconds: Double = 0.05) -> [Float] {
    let frame = max(1, sampleRate / 100), pad = Int(Double(sampleRate) * padSeconds)
    let floor = (speech.map(abs).max() ?? 0) * 0.01
    let loud = stride(from: 0, to: speech.count, by: frame).filter { start in
        let slice = speech[start..<min(start + frame, speech.count)]
        return (slice.reduce(0) { $0 + $1 * $1 } / Float(slice.count)).squareRoot() > floor
    }
    guard floor > 0, let first = loud.first, let last = loud.last else { return [] }
    return Array(speech[max(0, first - pad)..<min(speech.count, last + frame + pad)])
}
