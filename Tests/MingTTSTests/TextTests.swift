// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import MingTTS

/// Fixtures come from the Hugging Face tokenizer and the reference frontend
/// (scripts kept with the fixture: Fixtures/make_fixtures.py).
private struct Fixtures: Decodable {
    struct Tokenized: Decodable { var text: String; var ids: [Int32] }
    struct Prompt: Decodable { var text: String; var voice: String; var ids: [Int32]; var speaker_positions: [Int] }
    struct Split: Decodable { var text: String; var maxLength: Int; var fragments: [String]? }
    var tokenizer: [Tokenized]
    var prompt: [Prompt]
    var split: [Split]

    static let shared: Fixtures = {
        let url = Bundle.module.url(forResource: "text_fixtures", withExtension: "json", subdirectory: "Fixtures")!
        return try! JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: url))
    }()
}

private let tokenizer = testTokenizer

/// Swift String equality is canonical equivalence; the model sees code points.
private func scalars(_ s: String) -> [UInt32] { s.unicodeScalars.map(\.value) }

@Test(.enabled(if: weightsDirectory != nil, "set MING_WEIGHTS")) func textTokenizerMatchesHuggingFace() {
    let cases = Fixtures.shared.tokenizer
    let failures = cases.filter { tokenizer.encode($0.text) != $0.ids }
    for failure in failures.prefix(5) {
        Issue.record("\(failure.text.debugDescription.prefix(200)): got \(tokenizer.encode(failure.text).prefix(40)) want \(failure.ids.prefix(40))")
    }
    #expect(failures.isEmpty, "\(failures.count) of \(cases.count) strings differ")
}

@Test(.enabled(if: weightsDirectory != nil && goldensDirectory != nil, "set MING_WEIGHTS and MING_GOLDENS")) func textTokenizerMatchesReferenceGoldens() throws {
    let data = try Data(contentsOf: goldensDirectory!.appending(path: "tokenizer_cases.json"))
    let json = try JSONSerialization.jsonObject(with: data)
    let root = try #require(json as? [String: Any])
    let cases = try #require(root["cases"] as? [[String: Any]])
    for entry in cases {
        let text = try #require(entry["text"] as? String), ids = try #require(entry["ids"] as? [Int])
        #expect(tokenizer.encode(text) == ids.map(Int32.init), "\(text.debugDescription.prefix(120))")
    }
    let prompts = try #require(root["prompts"] as? [[String: Any]])
    for entry in prompts {
        let text = try #require(entry["text_prepared"] as? String), voice = try #require(entry["voice"] as? String)
        let prompt = buildPrompt(fragment: text, request: try ipRequest(text, voice), tokenizer: tokenizer)
        #expect(prompt.ids == (try #require(entry["input_ids"] as? [Int])).map(Int32.init), "\(text)")
        #expect(prompt.speakerPositions == (try #require(entry["speaker_positions"] as? [Int])))
    }
    #expect(!cases.isEmpty && !prompts.isEmpty)
}

/// The request the reference goldens were made from: task IP, English, a zero speaker slot.
private func ipRequest(_ text: String, _ voice: String) throws -> MingRequest {
    try MingRequest(text: text, task: .builtInVoice(MingVoice(ip: voice)), language: .english, conditioning: .zeroSpeaker)
}

@Test(.enabled(if: weightsDirectory != nil, "set MING_WEIGHTS")) func textPromptMatchesReferenceIPLayout() throws {
    for fixture in Fixtures.shared.prompt {
        let request = try ipRequest(fixture.text, fixture.voice)
        let prompt = buildPrompt(fragment: request.text, request: request, tokenizer: tokenizer)
        #expect(prompt.ids == fixture.ids, "\(fixture.text.prefix(80))")
        #expect(prompt.speakerPositions == fixture.speaker_positions)
    }
}

@Test(.enabled(if: weightsDirectory != nil, "set MING_WEIGHTS")) func textPromptSpeakerSlotIsTheVisionPad() throws {
    let prompt = buildPrompt(fragment: "Hello.", request: try ipRequest("Hello.", "US_Emma"), tokenizer: tokenizer)
    #expect(prompt.speakerPositions.map { prompt.ids[$0] } == [151654])
    #expect(prompt.ids.last == 151706)
}

@Test func textSplitMatchesReferenceFrontend() {
    func split(_ c: Fixtures.Split) -> [String] {
        splitText(prepareText(c.text, task: .builtInVoice(.emma), language: .english, normalizing: false), maxLength: c.maxLength)
    }
    let cases = Fixtures.shared.split
    let failures = cases.filter { scalars(split($0).joined(separator: "\u{1}")) != scalars(($0.fragments ?? []).joined(separator: "\u{1}")) }
    for failure in failures.prefix(5) {
        Issue.record("\(failure.text.debugDescription.prefix(200)) max \(failure.maxLength): got \(split(failure)) want \(failure.fragments ?? [])")
    }
    #expect(failures.isEmpty, "\(failures.count) of \(cases.count) splits differ")
}

@Test(.enabled(if: weightsDirectory != nil, "set MING_WEIGHTS")) func textTokenizerIsFastOnABulletin() {
    let bulletin = Fixtures.shared.tokenizer.map(\.text).filter { $0.split(separator: " ").count > 300 }.first!
    _ = tokenizer.encode(bulletin)
    let clock = ContinuousClock()
    let runs = 20
    let elapsed = clock.measure { for _ in 0..<runs { _ = tokenizer.encode(bulletin) } }
    let perRun = elapsed / runs
    print("encode \(bulletin.split(separator: " ").count) words: \(perRun)")
    #expect(perRun < .milliseconds(5))
}
