// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import XCTest
@testable import MingTTS

/// The CPU voice path (resample, Kaldi fbank, CAM++) against the cloning goldens. Runs when
/// MING_GOLDENS points at the goldens directory and MING_WEIGHTS at a directory holding campplus.onnx.
final class VoiceTests: XCTestCase {
    struct Golden {
        let sourceRate: Int
        let tensors: [String: [String: Any]]
        let url: URL

        init(_ name: String) throws {
            guard let goldens = ProcessInfo.processInfo.environment["MING_GOLDENS"] else {
                throw XCTSkip("set MING_GOLDENS to run the voice golden tests")
            }
            url = URL(fileURLWithPath: goldens).appendingPathComponent(name)
            let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: url.appendingPathComponent("manifest.json"))) as! [String: Any]
            sourceRate = manifest["prompt_source_rate"] as! Int
            tensors = manifest["tensors"] as! [String: [String: Any]]
        }

        func tensor(_ name: String) throws -> [Float] {
            let data = try Data(contentsOf: url.appendingPathComponent(tensors[name]!["file"] as! String))
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
    }

    static let cases = ["clone_george", "clone_lingguang"]

    func campplus() throws -> CAMPlus {
        guard let weights = ProcessInfo.processInfo.environment["MING_WEIGHTS"] else {
            throw XCTSkip("set MING_WEIGHTS to a directory holding campplus.onnx")
        }
        return try CAMPlus(onnx: URL(fileURLWithPath: weights).appendingPathComponent("campplus.onnx"))
    }

    func maxAbsError(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map { abs($0 - $1) }.max() ?? .infinity
    }

    func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for (x, y) in zip(a, b) { dot += Double(x) * Double(y); na += Double(x) * Double(x); nb += Double(y) * Double(y) }
        return dot / (na * nb).squareRoot()
    }

    func testResampleMatchesTorchaudio() throws {
        for name in Self.cases {
            let g = try Golden(name)
            let source = try g.tensor("prompt_source")
            for (rate, golden) in [(16000, "prompt_wave_16k"), (44100, "prompt_wave_44k")] {
                let want = try g.tensor(golden)
                let got = resample(source, from: g.sourceRate, to: rate)
                XCTAssertEqual(got.count, want.count, "\(name) \(golden) length")
                let error = maxAbsError(got, want)
                print("\(name) \(golden): max|err| \(error)")
                XCTAssertLessThanOrEqual(error, 4e-7, "\(name) \(golden) max abs error within 4e-7")
            }
        }
    }

    func testFbankMatchesKaldi() throws {
        for name in Self.cases {
            let g = try Golden(name)
            let want = try g.tensor("fbank")
            let (got, frames) = kaldiFbank(try g.tensor("prompt_wave_16k"))
            XCTAssertEqual(frames * 80, want.count, "\(name) fbank frame count")
            let error = maxAbsError(got, want)
            print("\(name) fbank: frames \(frames) max|err| \(error)")
            XCTAssertLessThanOrEqual(error, 1e-3, "\(name) fbank max abs error within 1e-3 (log domain)")
        }
    }

    func testEmbeddingMatchesONNX() throws {
        let model = try campplus()
        for name in Self.cases {
            let g = try Golden(name)
            let fbank = try g.tensor("fbank")
            let want = try g.tensor("spk_emb")
            let got = model.embed(fbank: fbank, frames: fbank.count / 80)
            XCTAssertEqual(got.count, 192, "\(name) embedding width")
            let (cos, error) = (cosine(got, want), maxAbsError(got, want))
            print("\(name) spk_emb from golden fbank: cosine \(cos) max|err| \(error)")
            XCTAssertGreaterThanOrEqual(cos, 0.9999, "\(name) embedding cosine >= 0.9999")
            XCTAssertLessThanOrEqual(error, 1e-3, "\(name) embedding max abs error within 1e-3")
        }
    }

    func testSourceToEmbeddingChain() throws {
        let model = try campplus()
        for name in Self.cases {
            let g = try Golden(name)
            let (fbank, frames) = kaldiFbank(resample(try g.tensor("prompt_source"), from: g.sourceRate, to: 16000))
            let cos = cosine(model.embed(fbank: fbank, frames: frames), try g.tensor("spk_emb"))
            print("\(name) spk_emb from source: cosine \(cos)")
            XCTAssertGreaterThanOrEqual(cos, 0.9999, "\(name) source -> embedding cosine >= 0.9999")
        }
    }

    /// Wall time per stage on a 24 kHz clip named by MING_VOICE_TIMING_WAV; prints, asserts nothing.
    func testStageTiming() throws {
        guard let path = ProcessInfo.processInfo.environment["MING_VOICE_TIMING_WAV"] else {
            throw XCTSkip("set MING_VOICE_TIMING_WAV to time the voice path")
        }
        let model = try campplus()
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
        let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: buffer)
        let wave = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
        let rate = Int(file.processingFormat.sampleRate)
        func time<T>(_ body: () -> T) -> (T, Double) {
            let start = DispatchTime.now().uptimeNanoseconds
            let value = body()
            return (value, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        }
        let (wave16k, resampleMs) = time { resample(wave, from: rate, to: 16000) }
        let (_, resample44Ms) = time { resample(wave, from: rate, to: 44100) }
        let ((fbank, frames), fbankMs) = time { kaldiFbank(wave16k) }
        let (_, embedMs) = time { model.embed(fbank: fbank, frames: frames) }
        print(String(format: "%.2f s at %d Hz: resample16k %.1f ms, resample44k %.1f ms, fbank %.1f ms (%d frames), embed %.1f ms",
                     Double(wave.count) / Double(rate), rate, resampleMs, resample44Ms, fbankMs, frames, embedMs))
    }
}

final class TrimSilenceTests: XCTestCase {
    func testKeepsSpeechWithFiftyMillisecondsEitherSide() {
        let rate = 1000
        let tone = (0..<500).map { Float(sin(Double($0) * 0.3)) * 0.5 }
        let hiss = [Float](repeating: 0.001, count: 400)
        let trimmed = trimSilence(hiss + tone + hiss, sampleRate: rate)
        XCTAssertEqual(trimmed.count, 50 + 500 + 50)
        XCTAssertEqual(Array(trimmed[50..<550]), tone)
    }

    func testSilenceThroughoutKeepsNothing() {
        XCTAssertEqual(trimSilence([Float](repeating: 0, count: 2000), sampleRate: 1000), [])
    }
}
