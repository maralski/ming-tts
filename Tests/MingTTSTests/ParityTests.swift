// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import MingTTS

/// Teacher-forced parity against the reference goldens. Runs when MING_GOLDENS points at the
/// goldens directory and MING_WEIGHTS at the checkpoint; skips otherwise.
final class ParityTests: XCTestCase {
    func testEmmaShortTeacherForced() throws {
        let env = ProcessInfo.processInfo.environment
        guard let goldens = env["MING_GOLDENS"], let weights = env["MING_WEIGHTS"] else {
            throw XCTSkip("set MING_GOLDENS and MING_WEIGHTS to run the parity test")
        }
        let caseURL = URL(fileURLWithPath: goldens).appendingPathComponent("emma_short")
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: caseURL.appendingPathComponent("manifest.json"))) as! [String: Any]
        let tensors = manifest["tensors"] as! [String: [String: Any]]
        func golden(_ name: String) throws -> [Float] {
            let data = try Data(contentsOf: caseURL.appendingPathComponent(tensors[name]!["file"] as! String))
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        let steps = manifest["steps"] as! Int
        let latents = try golden("latents")
        let latentRows = stride(from: 0, to: latents.count, by: 64).map { Array(latents[$0..<($0 + 64)]) }
        let model = try MingTTSModel(directory: URL(fileURLWithPath: weights))
        let run = try model.parityRun(promptEmbeddings: try golden("inputs_embeds"), noise: try golden("noise"), steps: steps,
                                      cfg: Float(manifest["cfg"] as! Double), teacherLatents: latentRows)
        func cosine(_ a: [Float], _ b: [Float]) -> Double {
            XCTAssertEqual(a.count, b.count)
            var dot = 0.0, na = 0.0, nb = 0.0
            for i in 0..<min(a.count, b.count) {
                dot += Double(a[i]) * Double(b[i]); na += Double(a[i]) * Double(a[i]); nb += Double(b[i]) * Double(b[i])
            }
            return dot / (na * nb).squareRoot()
        }
        let c = run.capture
        let checks: [(String, [Float])] = [
            ("llm_layer0_prefill", c.llmLayer0Prefill), ("llm_layer23_prefill", c.llmLayer23Prefill),
            ("llm_hidden", Array(c.llmHidden.joined())), ("dit_velocity_step0", c.velocityStep0),
            ("latents", Array(c.latents.joined())), ("agg_out", Array(c.aggOut.joined())),
            ("vae_hidden", Array(c.vaeHidden.joined())), ("waveform_stream", run.pcm),
        ]
        for (name, got) in checks {
            XCTAssertGreaterThanOrEqual(cosine(got, try golden(name)), 0.9999, name)
        }
        let stop = try golden("stop_prob")
        XCTAssertEqual(c.stopProb.count, stop.count)
        XCTAssertLessThan(zip(c.stopProb, stop).map { abs($0 - $1) }.max() ?? 1, 1e-3, "stop_prob")
        XCTAssertEqual(run.report.steps, steps)
    }
}
