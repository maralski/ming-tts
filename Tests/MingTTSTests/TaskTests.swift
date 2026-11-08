// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import MingTTS

/// Every task mode against the reference: fragments and prompt ids per fragment, from
/// Fixtures/make_task_fixtures.py.
private struct TaskCase: Decodable {
    var task: String
    var controls: [String: String]
    var instruction: String?
    var language: String?
    var speaker: String
    var text: String
    var fragments: [String]
    var prompts: [Prompt]
    struct Prompt: Decodable { var ids: [Int32]; var speaker_positions: [Int] }

    var request: MingRequest {
        get throws {
            let task: MingTask = switch self.task {
            case "TTS", "MULTILINGUAL": .plain
            case "IP": .builtInVoice(MingVoice(ip: controls["ip"]!))
            case "VOICE_DESIGN": .voiceDesign(controls["style"]!)
            case "FREEFORM": .freeform(instruction!)
            case "EMOTION": .emotion(MingEmotion(rawValue: controls["emotion"]!)!)
            case "BASIC": .attributes(speed: controls["speed"].map(level), pitch: controls["pitch"].map(level), volume: controls["volume"].map(level))
            case "DIALECT": .dialect(MingDialect(rawValue: controls["dialect"]!)!)
            default: fatalError("unknown task \(self.task)")
            }
            return try MingRequest(text: text, task: task, language: language.map { MingLanguage(id: $0)! }, conditioning: speaker == "zero" ? .zeroSpeaker : .noSpeaker)
        }
    }
}

private func level(_ value: String) -> MingLevel {
    MingLevel.allCases.first { $0.speed == value || $0.pitchOrVolume == value }!
}

private let cases: [TaskCase] = {
    let url = Bundle.module.url(forResource: "task_fixtures", withExtension: "json", subdirectory: "Fixtures")!
    return try! JSONDecoder().decode([TaskCase].self, from: Data(contentsOf: url))
}()

private let tokenizer = testTokenizer

private func label(_ c: TaskCase) -> String { "\(c.task) \(c.controls) \(c.language ?? "-") \(c.speaker): \(c.text.prefix(40).debugDescription)" }

/// Reviewed differences, reference fragment to Swift fragment. The reference's TalkerPreProcessor
/// deletes the space between punctuation and a tag (its Chinese-space rule, which normalizeEnglish
/// leaves out because it glues English words to symbols), and its English grammar then pads "(".
private let deviations = [
    "[Laughter] That was funny, really[Sigh]honestly. Well [Uhm] I think [Breathing] so?[Question-huh]":
        "[Laughter] That was funny, really[Sigh]honestly. Well [Uhm] I think [Breathing] so? [Question-huh]",
    "Notice Bold news: see the site and code [Cough]( aside) now.": "Notice Bold news: see the site and code [Cough](aside) now.",
]

@Test func taskFragmentsMatchReference() throws {
    var failures = 0
    for c in cases {
        let got = try prepareFragments(try c.request, maxLength: 24)
        let want = c.fragments.map { deviations[$0] ?? $0 }
        if got.map({ Array($0.unicodeScalars) }) != want.map({ Array($0.unicodeScalars) }) {
            failures += 1
            if failures <= 8 { Issue.record("\(label(c))\n got  \(got)\n want \(want)") }
        }
    }
    #expect(failures == 0, "\(failures) of \(cases.count) cases differ")
    let fragments = Set(cases.flatMap(\.fragments))
    #expect(deviations.keys.allSatisfy(fragments.contains), "a reviewed deviation no longer occurs")
}

@Test(.enabled(if: weightsDirectory != nil, "set MING_WEIGHTS")) func taskPromptsMatchReference() throws {
    var failures = 0
    for c in cases {
        let request = try c.request
        let got = c.fragments.map { buildPrompt(fragment: $0, request: request, tokenizer: tokenizer) }
        if got.map(\.ids) != c.prompts.map(\.ids) || got.map(\.speakerPositions) != c.prompts.map(\.speaker_positions) {
            failures += 1
            if failures <= 8 { Issue.record("\(label(c))\n got  \(got.map(\.ids))\n want \(c.prompts.map(\.ids))") }
        }
    }
    #expect(failures == 0, "\(failures) of \(cases.count) cases differ")
    #expect(Set(cases.map(\.task)).count == 8 && Set(cases.map(\.speaker)).count == 2)
}

@Test func taskRequestRejectsWhatTheReferenceRejects() {
    #expect(throws: MingTTSError.self) { try MingRequest(text: " \n ", task: .plain, conditioning: .zeroSpeaker) }
    #expect(throws: MingTTSError.self) { try MingRequest(text: "Hi.", task: .attributes(speed: nil, pitch: nil, volume: nil), conditioning: .zeroSpeaker) }
    #expect(throws: MingTTSError.self) { try MingRequest(text: "Hi.", task: .freeform("  "), conditioning: .zeroSpeaker) }
    #expect(throws: MingTTSError.self) { try MingRequest(text: "Hi.", task: .voiceDesign(""), conditioning: .zeroSpeaker) }
    #expect(throws: MingTTSError.self) { try MingRequest(text: "Hi.", task: .dialect(.cantonese), language: .chinese, conditioning: .zeroSpeaker) }
    #expect(throws: MingTTSError.self) { try prepareFragments(try MingRequest(text: "!!!", task: .plain, conditioning: .zeroSpeaker), maxLength: 24) }
}

@Test func taskStyleTextSplitsLikeAnyOther() throws {
    let text = "Officials in the coastal city said on Tuesday that the new flood barrier had held through the overnight storm. Engineers will inspect the structure this week."
    #expect(try prepareFragments(try MingRequest(text: text, task: .freeform("Calm."), language: .english, conditioning: .zeroSpeaker), maxLength: 24).count == 2)
    #expect(try prepareFragments(try MingRequest(text: text, task: .plain, language: .english, conditioning: .zeroSpeaker), maxLength: 24).count == 2)
}

/// Clone prompts across the task modes against the reference contract, from
/// Fixtures/make_clone_fixtures.py: the transcript block before the text, one `<audioPatch>` row
/// per reference latent after `<audio>`, and the speaker row only with the embedding.
private struct CloneCase: Decodable {
    var task: String
    var controls: [String: String]
    var instruction: String?
    var language: String?
    var text: String
    var prompt_text: String
    var latent_steps: Int
    var speaker_embedding: Bool
    var ids: [Int32]
    var speaker_positions: [Int]
    var latent_position: Int
}

@Test(.enabled(if: weightsDirectory != nil, "set MING_WEIGHTS")) func clonePromptsMatchReference() throws {
    let url = Bundle.module.url(forResource: "clone_fixtures", withExtension: "json", subdirectory: "Fixtures")!
    let cases = try JSONDecoder().decode([CloneCase].self, from: Data(contentsOf: url))
    var failures = 0
    for c in cases {
        let voice = try MingReferenceVoice(transcript: c.prompt_text, latents: [Float](repeating: 0.5, count: 64 * c.latent_steps),
                                           speakerEmbedding: [Float](repeating: 0.1, count: 192))
        let fixture = TaskCase(task: c.task, controls: c.controls, instruction: c.instruction, language: c.language, speaker: "none",
                               text: c.text, fragments: [], prompts: [])
        let request = try fixture.request.conditioned(.clone(voice, speakerEmbedding: c.speaker_embedding))
        let got = buildPrompt(fragment: c.text, request: request, tokenizer: tokenizer)
        if got.ids != c.ids || got.speakerPositions != c.speaker_positions || (got.latentPosition ?? -1) != c.latent_position {
            failures += 1
            if failures <= 5 { Issue.record("\(c.task) \(c.controls) spk \(c.speaker_embedding) steps \(c.latent_steps): \(c.text.prefix(30).debugDescription)\n got  \(got.ids) \(got.speakerPositions) \(got.latentPosition ?? -1)\n want \(c.ids) \(c.speaker_positions) \(c.latent_position)") }
        }
    }
    #expect(failures == 0, "\(failures) of \(cases.count) cases differ")
    #expect(Set(cases.map(\.task)).count == 8 && cases.contains { !$0.speaker_embedding } && cases.contains { $0.latent_steps == 1 })
}

@Test func referenceVoiceRejectsBadShapes() {
    #expect(throws: MingTTSError.self) { try MingReferenceVoice(transcript: " ", latents: [Float](repeating: 0, count: 64), speakerEmbedding: [Float](repeating: 0, count: 192)) }
    #expect(throws: MingTTSError.self) { try MingReferenceVoice(transcript: "Hi.", latents: [Float](repeating: 0, count: 63), speakerEmbedding: [Float](repeating: 0, count: 192)) }
    #expect(throws: MingTTSError.self) { try MingReferenceVoice(transcript: "Hi.", latents: [Float](repeating: 0, count: 64), speakerEmbedding: [Float](repeating: 0, count: 191)) }
}
