// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import MingTTS

@Test func languageCodesWithAScriptOrRegionNameTheirBaseLanguage() {
    #expect(MingLanguage(id: "zh-Hans") == .chinese)
    #expect(MingLanguage(id: "zh-Hant") == .chinese)
    #expect(MingLanguage(id: "zh_TW") == .chinese)
    #expect(MingLanguage(id: "en-US") == .english)
    #expect(MingLanguage(id: "es-MX") == .spanish)
    #expect(MingLanguage(id: "JA") == .japanese)
    #expect(MingLanguage(id: "Korean") == .korean)
    #expect(MingLanguage(id: "yue") == .chinese)
    #expect(MingLanguage(id: "it") == nil)
    #expect(MingLanguage(id: "it-IT") == nil)
    #expect(MingLanguage(id: "") == nil)
}

@Test func soundTagsAreTheFourteenTheFrontendKeeps() {
    #expect(MingSoundTag.allCases.count == 14)
    #expect(MingSoundTag.laughter.rawValue == "[Laughter]")
    for tag in MingSoundTag.allCases {
        #expect(stripMarkdown("a \(tag.rawValue)(b) c") == "a \(tag.rawValue)(b) c")
    }
    #expect(stripMarkdown("a [Laughs](b) c") == "a Laughs c")
}

@Test func catalogVoicesSayTheirKindLanguageAndEmotion() {
    let kinds = Dictionary(grouping: MingVoice.catalog, by: \.kind).mapValues(\.count)
    #expect(kinds == [.chinese: 33, .chineseEmotion: 197, .chineseCharacter: 8, .english: 3])
    #expect(MingVoice.emma.kind == .english)
    #expect(MingVoice.emma.language == .english)
    #expect(MingVoice.emma.emotion == nil)
    #expect(MingVoice(ip: "灵妩月_高兴").emotion == .happy)
    #expect(MingVoice(ip: "灵妩月_生气").emotion == .angry)
    #expect(MingVoice(ip: "灵妩月_生气").language == .chinese)
    #expect(MingVoice(ip: "灵小安").kind == .chinese)
    let emotions = Set(MingVoice.catalog.compactMap(\.emotion))
    #expect(emotions == Set(MingEmotion.allCases))
}

@Test func checkpointNamesTheFilesAFolderLacks() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("ming-checkpoint-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let checkpoint = MingCheckpoint(directory: folder)
    #expect(checkpoint.missingFiles == ["model.safetensors", "tokenizer.json", "campplus.onnx"])
    try Data().write(to: folder.appendingPathComponent("tokenizer.json"))
    #expect(checkpoint.missingFiles == ["model.safetensors", "campplus.onnx"])
    #expect(!checkpoint.isComplete)
    do {
        _ = try MingTTSModel(directory: folder)
        Issue.record("loading a folder without the weights did not throw")
    } catch let error as MingTTSError {
        #expect(error.description == "bad checkpoint: \(folder.path) lacks model.safetensors")
    }
}

@Test func checkpointIsFoundByEnvironmentThenInTheHuggingFaceCache() throws {
    let cache = FileManager.default.temporaryDirectory.appendingPathComponent("ming-cache-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: cache) }
    #expect(MingCheckpoint.find(environment: ["HF_HUB_CACHE": cache.path]) == nil)
    let snapshots = cache.appendingPathComponent("models--inclusionAI--Ming-3.0-tts-0.5B/snapshots")
    let partial = snapshots.appendingPathComponent("newer"), complete = snapshots.appendingPathComponent("older")
    for (folder, files, age) in [(complete, MingCheckpoint.synthesisFiles + MingCheckpoint.cloningFiles, -100.0), (partial, ["tokenizer.json"], 0.0)] {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for file in files { try Data().write(to: folder.appendingPathComponent(file)) }
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(age)], ofItemAtPath: folder.path)
    }
    #expect(MingCheckpoint.find(environment: ["HF_HUB_CACHE": cache.path])?.directory.lastPathComponent == "older")
    #expect(MingCheckpoint.find(environment: ["HF_HOME": "/nonexistent", "HF_HUB_CACHE": cache.path])?.isComplete == true)
    try FileManager.default.removeItem(at: complete)
    #expect(MingCheckpoint.find(environment: ["HF_HUB_CACHE": cache.path])?.directory.lastPathComponent == "newer")
    #expect(MingCheckpoint.find(environment: ["MING_WEIGHTS": "/somewhere/else", "HF_HUB_CACHE": cache.path])?.directory.path == "/somewhere/else")
}

@Test func referenceVoiceReportsItsLength() throws {
    let voice = try MingReferenceVoice(transcript: "Hello.", latents: [Float](repeating: 0, count: 64 * 125),
                                       speakerEmbedding: [Float](repeating: 0, count: 192))
    #expect(voice.seconds == 10)
    #expect(voice.seconds < MingReferenceVoice.recommendedMaxSeconds)
}
