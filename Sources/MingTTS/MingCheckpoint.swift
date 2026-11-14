// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A folder that should hold the Ming-3.0-tts-0.5B checkpoint. Check it before loading, so a
/// partial download is reported by the names of the files it lacks.
public struct MingCheckpoint: Sendable, Equatable {
    /// The Hugging Face repository the checkpoint is published in.
    public static let repository = "inclusionAI/Ming-3.0-tts-0.5B"
    /// The files `MingTTSModel` reads to speak.
    public static let synthesisFiles = ["model.safetensors", "tokenizer.json"]
    /// The file `MingTTSModel.encodeReference` reads as well, to clone a voice.
    public static let cloningFiles = ["campplus.onnx"]

    public let directory: URL
    public init(directory: URL) { self.directory = directory }

    /// The files of `synthesisFiles` and `cloningFiles` the folder lacks, in that order.
    public var missingFiles: [String] { missing(Self.synthesisFiles + Self.cloningFiles) }
    public var isComplete: Bool { missingFiles.isEmpty }

    func missing(_ files: [String]) -> [String] {
        files.filter { !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }

    /// - Throws: `MingTTSError.badCheckpoint` naming the folder and each of `files` it lacks.
    func require(_ files: [String]) throws {
        let missing = missing(files)
        guard missing.isEmpty else {
            throw MingTTSError.badCheckpoint("\(directory.path) lacks \(missing.joined(separator: ", "))")
        }
    }

    /// The folder `MING_WEIGHTS` names, complete or not, so a caller can say what it lacks. Else
    /// the newest complete snapshot of `repository` in the Hugging Face cache, or the newest one
    /// when none is complete. Nil when there is neither.
    public static func find(environment: [String: String] = ProcessInfo.processInfo.environment) -> MingCheckpoint? {
        if let path = environment["MING_WEIGHTS"], !path.isEmpty {
            return MingCheckpoint(directory: URL(fileURLWithPath: path, isDirectory: true))
        }
        let snapshots = huggingFaceCache(environment)
            .appendingPathComponent("models--" + repository.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent("snapshots")
        let keys: Set<URLResourceKey> = [.contentModificationDateKey, .isDirectoryKey]
        let found = (try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: Array(keys))) ?? []
        let newestFirst = found
            .filter { (try? $0.resourceValues(forKeys: keys).isDirectory) == true }
            .sorted { modified($0) > modified($1) }
            .map(MingCheckpoint.init(directory:))
        return newestFirst.first(where: \.isComplete) ?? newestFirst.first
    }

    /// `HF_HUB_CACHE`, else `hub` in `HF_HOME`, else `~/.cache/huggingface/hub`, as `huggingface_hub` resolves it.
    static func huggingFaceCache(_ environment: [String: String]) -> URL {
        if let cache = environment["HF_HUB_CACHE"], !cache.isEmpty { return URL(fileURLWithPath: cache, isDirectory: true) }
        let home = environment["HF_HOME"].flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0, isDirectory: true) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface")
        return home.appendingPathComponent("hub")
    }

    private static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }
}
