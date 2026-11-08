// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import MingTTS

/// Zero-shot cloning against the reference goldens, stage by stage: the prompt contract, the
/// spliced prompt embeddings, the VAE encoder's posterior, the whole encodeReference chain, and a
/// teacher-forced generation from the cloned prompt. Runs when MING_GOLDENS points at the goldens
/// directory and MING_WEIGHTS at the checkpoint; skips otherwise.
final class CloneTests: XCTestCase {
    struct Golden {
        let url: URL
        let manifest: [String: Any]
        var tensors: [String: [String: Any]] { manifest["tensors"] as! [String: [String: Any]] }

        init(_ name: String) throws {
            guard let goldens = ProcessInfo.processInfo.environment["MING_GOLDENS"] else {
                throw XCTSkip("set MING_GOLDENS and MING_WEIGHTS to run the clone golden tests")
            }
            url = URL(fileURLWithPath: goldens).appendingPathComponent(name)
            manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: url.appendingPathComponent("manifest.json"))) as! [String: Any]
        }

        func tensor(_ name: String) throws -> [Float] {
            let data = try Data(contentsOf: url.appendingPathComponent(tensors[name]!["file"] as! String))
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }

        func voice() throws -> MingReferenceVoice {
            try MingReferenceVoice(transcript: manifest["prompt_text"] as! String, latents: try tensor("prompt_latent"), speakerEmbedding: try tensor("spk_emb"))
        }

        func request() throws -> MingRequest {
            try MingRequest(text: manifest["text"] as! String, task: .plain,
                            language: (manifest["language_id"] as? String).flatMap(MingLanguage.init(id:)), conditioning: .clone(try voice()))
        }
    }

    static let cases = ["clone_george", "clone_lingguang"]
    nonisolated(unsafe) static var sharedModel: MingTTSModel?

    func model() throws -> MingTTSModel {
        if let m = Self.sharedModel { return m }
        guard let weights = ProcessInfo.processInfo.environment["MING_WEIGHTS"] else { throw XCTSkip("set MING_WEIGHTS to run the clone golden tests") }
        let m = try MingTTSModel(directory: URL(fileURLWithPath: weights))
        Self.sharedModel = m
        return m
    }

    func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<min(a.count, b.count) {
            dot += Double(a[i]) * Double(b[i]); na += Double(a[i]) * Double(a[i]); nb += Double(b[i]) * Double(b[i])
        }
        return dot / max((na * nb).squareRoot(), 1e-30)
    }

    func maxAbs(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).map { abs($0 - $1) }.max() ?? .infinity }

    func testClonePromptIdsMatchGolden() throws {
        let model = try model()
        for name in Self.cases {
            let g = try Golden(name)
            let prompt = model.prompt(fragment: g.manifest["text"] as! String, of: try g.request())
            XCTAssertEqual(prompt.ids, (g.manifest["input_ids"] as! [Int]).map(Int32.init), "\(name) ids")
            XCTAssertEqual(prompt.speakerPositions, g.manifest["speaker_positions"] as! [Int], "\(name) speaker rows")
            XCTAssertEqual(prompt.latentPosition, g.manifest["latent_position"] as! Int, "\(name) first latent row")
        }
    }

    /// The speaker row is spk_head of the embedding and each latent row the aggregator's output.
    func testClonePromptEmbeddingsMatchGolden() throws {
        let model = try model()
        for name in Self.cases {
            let g = try Golden(name)
            let prompt = model.prompt(fragment: g.manifest["text"] as! String, of: try g.request())
            let got = try model.promptEmbeddings(prompt)
            let want = try g.tensor("inputs_embeds")
            XCTAssertEqual(got.count, want.count, "\(name) rows")
            let hidden = 896
            let speaker = prompt.speakerPositions[0], latent = prompt.latentPosition!
            let rows = try g.voice().latentRows
            XCTAssertGreaterThanOrEqual(cosine(Array(got[(speaker * hidden)..<((speaker + 1) * hidden)]), try g.tensor("spk_row")), 0.9999, "\(name) speaker row")
            XCTAssertLessThan(maxAbs(Array(got[(speaker * hidden)..<((speaker + 1) * hidden)]), try g.tensor("spk_row")), 1e-3, "\(name) speaker row max abs")
            let latentRange = (latent * hidden)..<((latent + rows) * hidden)
            XCTAssertGreaterThanOrEqual(cosine(Array(got[latentRange]), Array(want[latentRange])), 0.9999, "\(name) latent rows (aggregator)")
            XCTAssertGreaterThanOrEqual(cosine(got, want), 0.9999, "\(name) inputs_embeds")
        }
    }

    /// The Metal encoder's posterior from the golden padded waveform, and the sample with the golden draw.
    func testEncoderPosteriorMatchesGolden() throws {
        let model = try model()
        for name in Self.cases {
            let g = try Golden(name)
            let (mean, std) = try model.encodeLatents(samples: try g.tensor("prompt_wave_padded"))
            let wantMean = try g.tensor("enc_mean"), wantStd = try g.tensor("enc_std")
            XCTAssertEqual(mean.count, wantMean.count, "\(name) latent count")
            // Measured 2.5e-5 on means of scale 3; a 64-key window moves them by 0.056 and cosine misses that.
            XCTAssertLessThan(maxAbs(mean, wantMean), 1e-3, "\(name) enc_mean max abs")
            // The stds are 1e-4..0.018, so the sample is nearly the mean; measured 1.1e-5 relative, exp in
            // place of softplus gives 1e-3..9e-3.
            XCTAssertLessThan(zip(std, wantStd).map { abs($0 - $1) / $1 }.max()!, 5e-4, "\(name) enc_std max relative error")
            let noise = try g.tensor("enc_noise")
            let latent = (0..<mean.count).map { mean[$0] + std[$0] * noise[$0] }
            XCTAssertGreaterThanOrEqual(cosine(latent, try g.tensor("prompt_latent")), 0.9999, "\(name) prompt_latent cosine")
        }
    }

    /// From the source clip through resample, fbank, CAM++, padding and the encoder.
    func testEncodeReferenceChainMatchesGolden() throws {
        let model = try model()
        for name in Self.cases {
            let g = try Golden(name)
            let voice = try model.encodeReference(samples: try g.tensor("prompt_source"), sampleRate: g.manifest["prompt_source_rate"] as! Int,
                                                  transcript: g.manifest["prompt_text"] as! String, noise: try g.tensor("enc_noise"), seed: 0)
            XCTAssertEqual(voice.latentRows, g.manifest["prompt_latent_steps"] as! Int, "\(name) latent rows")
            XCTAssertGreaterThanOrEqual(cosine(voice.speakerEmbedding, try g.tensor("spk_emb")), 0.9999, "\(name) speaker embedding")
            XCTAssertGreaterThanOrEqual(cosine(voice.latents, try g.tensor("prompt_latent")), 0.9999, "\(name) latents from the clip")
        }
    }

    /// Encode time for a clip, cold (first use builds the encoder weights and parses campplus.onnx)
    /// and warm, split into the CPU voice path and the GPU encoder. Needs MING_CLONE_TIMING_WAV.
    func testEncodeReferenceTiming() throws {
        guard let wav = ProcessInfo.processInfo.environment["MING_CLONE_TIMING_WAV"] else { throw XCTSkip("set MING_CLONE_TIMING_WAV to time encodeReference") }
        let model = try model()
        let (samples, rate) = try MingTTSModel.readMono(URL(fileURLWithPath: wav))
        let seconds = Double(samples.count) / Double(rate)
        func timed<T>(_ body: () throws -> T) throws -> (T, Double) {
            let t0 = Date()
            let value = try body()
            return (value, Date().timeIntervalSince(t0) * 1000)
        }
        let (_, cold) = try timed { try model.encodeReference(samples: samples, sampleRate: rate, transcript: "timing") }
        var warm: [Double] = []
        for _ in 0..<3 { warm.append(try timed { try model.encodeReference(samples: samples, sampleRate: rate, transcript: "timing") }.1) }
        let (wave16k, tResample16) = try timed { resample(samples, from: rate, to: 16000) }
        let (fbank, tFbank) = try timed { kaldiFbank(wave16k) }
        let net = try CAMPlus(onnx: URL(fileURLWithPath: ProcessInfo.processInfo.environment["MING_WEIGHTS"]!).appendingPathComponent("campplus.onnx"))
        let (_, tEmbed) = try timed { net.embed(fbank: fbank.features, frames: fbank.frames) }
        let (wave44k, tResample44) = try timed { resample(samples, from: rate, to: 44100) }
        let align = 3528, remainder = wave44k.count % align
        let padded = remainder == 0 ? wave44k : [Float](repeating: 0, count: align - remainder) + wave44k
        let (_, tEncoder) = try timed { try model.encodeLatents(samples: padded) }
        print(String(format: "encodeReference %.2f s clip at %d Hz: cold %.0f ms, warm %@ ms; resample16k %.1f, fbank %.1f, CAM++ %.1f, resample44k %.1f, VAE encoder %.1f ms",
                     seconds, rate, cold, warm.map { String(format: "%.0f", $0) }.joined(separator: "/"), tResample16, tFbank, tEmbed, tResample44, tEncoder))
    }

    /// Teacher-forced generation from the golden cloned prompt, with the reference's last latent as
    /// the step 0 history; the same stage checks as ParityTests.
    func testCloneTeacherForcedParity() throws {
        let model = try model()
        for name in Self.cases {
            let g = try Golden(name)
            let steps = g.manifest["steps"] as! Int
            let latents = try g.tensor("latents")
            let latentRows = stride(from: 0, to: latents.count, by: 64).map { Array(latents[$0..<($0 + 64)]) }
            let run = try model.parityRun(promptEmbeddings: try g.tensor("inputs_embeds"), noise: try g.tensor("noise"), steps: steps,
                                          cfg: Float(g.manifest["cfg"] as! Double), teacherLatents: latentRows,
                                          initialHistory: MingTTSModel.initialHistory(try g.voice()))
            let c = run.capture
            let checks: [(String, [Float])] = [
                ("llm_layer0_prefill", c.llmLayer0Prefill), ("llm_layer23_prefill", c.llmLayer23Prefill),
                ("llm_hidden", Array(c.llmHidden.joined())), ("dit_velocity_step0", c.velocityStep0),
                ("latents", Array(c.latents.joined())), ("agg_out", Array(c.aggOut.joined())),
                ("vae_hidden", Array(c.vaeHidden.joined())), ("waveform_stream", run.pcm),
            ]
            for (tensor, got) in checks {
                let want = try g.tensor(tensor)
                XCTAssertEqual(got.count, want.count, "\(name) \(tensor) count")
                XCTAssertGreaterThanOrEqual(cosine(got, want), 0.9999, "\(name) \(tensor)")
            }
            let stop = try g.tensor("stop_prob")
            XCTAssertEqual(c.stopProb.count, stop.count, "\(name) steps")
            XCTAssertLessThan(zip(c.stopProb, stop).map { abs($0 - $1) }.max() ?? 1, 1e-3, "\(name) stop_prob")
            XCTAssertEqual(run.report.steps, steps, "\(name) stopped at the golden step")
        }
    }
}
