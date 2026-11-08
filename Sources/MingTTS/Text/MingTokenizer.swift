// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Qwen2 byte-level BPE, read from the checkpoint's tokenizer.json. Matches Hugging Face
/// `tokenizer.encode(s, add_special_tokens=False)`.
struct MingTokenizer: @unchecked Sendable {
    enum LoadError: Error {
        case malformed(String)
    }

    private struct Merge {
        var rank: Int32
        var id: Int32
    }

    private struct AddedToken {
        var bytes: [UInt8]
        var id: Int32
    }

    private let byteIDs: [Int32]
    private let merges: [UInt64: Merge]
    /// Matched on the raw text, longest first.
    private let rawAdded: [AddedToken]
    /// Matched on the NFC text between raw matches, longest first.
    private let normalizedAdded: [AddedToken]
    private let addedIDs: [String: Int32]
    private let preTokenizer: NSRegularExpression
    private let nfc: Bool

    init(tokenizerJSON: URL) throws {
        let data = try Data(contentsOf: tokenizerJSON)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              model["type"] as? String == "BPE",
              let vocab = model["vocab"] as? [String: Int],
              let mergeList = model["merges"] as? [Any],
              let added = root["added_tokens"] as? [[String: Any]]
        else { throw LoadError.malformed("not a BPE tokenizer.json") }

        let unicodeToByte = Dictionary(uniqueKeysWithValues: byteToUnicode.enumerated().map { ($1, UInt8($0)) })
        // Keyed by decoded bytes: Swift String keys compare by canonical equivalence.
        var idByBytes: [[UInt8]: Int32] = [:]
        idByBytes.reserveCapacity(vocab.count)
        for (token, id) in vocab {
            var bytes: [UInt8] = []
            for scalar in token.unicodeScalars {
                guard let byte = unicodeToByte[scalar] else { throw LoadError.malformed("vocab token \(token) is not byte-level") }
                bytes.append(byte)
            }
            idByBytes[bytes] = Int32(id)
        }
        guard idByBytes.count == vocab.count else { throw LoadError.malformed("vocab tokens collide") }
        byteIDs = try (0..<256).map { byte in
            guard let id = idByBytes[[UInt8(byte)]] else { throw LoadError.malformed("no vocab entry for byte \(byte)") }
            return id
        }

        func decode(_ token: String) throws -> [UInt8] {
            try token.unicodeScalars.map {
                guard let byte = unicodeToByte[$0] else { throw LoadError.malformed("merge token \(token) is not byte-level") }
                return byte
            }
        }
        var merges: [UInt64: Merge] = [:]
        merges.reserveCapacity(mergeList.count)
        for (rank, entry) in mergeList.enumerated() {
            let pair: [String]
            if let array = entry as? [String] { pair = array }
            else if let line = entry as? String { pair = line.split(separator: " ", omittingEmptySubsequences: false).map(String.init) }
            else { throw LoadError.malformed("merge \(rank)") }
            guard pair.count == 2 else { throw LoadError.malformed("merge \(rank)") }
            let left = try decode(pair[0]), right = try decode(pair[1])
            guard let a = idByBytes[left], let b = idByBytes[right], let merged = idByBytes[left + right]
            else { throw LoadError.malformed("merge \(rank) names unknown tokens") }
            merges[Self.key(a, b)] = Merge(rank: Int32(rank), id: merged)
        }
        self.merges = merges

        var raw: [AddedToken] = [], normalized: [AddedToken] = [], ids: [String: Int32] = [:]
        for token in added {
            guard let content = token["content"] as? String, let id = token["id"] as? Int else { throw LoadError.malformed("added token") }
            for flag in ["lstrip", "rstrip", "single_word"] where token[flag] as? Bool == true {
                throw LoadError.malformed("added token \(content) uses \(flag), which this tokenizer does not implement")
            }
            let entry = AddedToken(bytes: Array(content.utf8), id: Int32(id))
            if token["normalized"] as? Bool == true { normalized.append(entry) } else { raw.append(entry) }
            ids[content] = Int32(id)
        }
        rawAdded = raw.sorted { $0.bytes.count > $1.bytes.count }
        normalizedAdded = normalized.sorted { $0.bytes.count > $1.bytes.count }
        addedIDs = ids

        switch (root["normalizer"] as? [String: Any])?["type"] as? String {
        case nil: nfc = false
        case "NFC": nfc = true
        case let other?: throw LoadError.malformed("normalizer \(other)")
        }

        guard let pre = root["pre_tokenizer"] as? [String: Any],
              pre["type"] as? String == "Sequence",
              let steps = pre["pretokenizers"] as? [[String: Any]],
              steps.count == 2,
              steps[0]["type"] as? String == "Split",
              steps[0]["behavior"] as? String == "Isolated",
              steps[0]["invert"] as? Bool == false,
              let pattern = (steps[0]["pattern"] as? [String: Any])?["Regex"] as? String,
              steps[1]["type"] as? String == "ByteLevel",
              steps[1]["add_prefix_space"] as? Bool == false,
              steps[1]["use_regex"] as? Bool == false
        else { throw LoadError.malformed("pre_tokenizer is not the Qwen2 split + byte-level sequence") }
        preTokenizer = try NSRegularExpression(pattern: pattern)
    }

    /// The id of an added token such as `<audio>`, or nil when the tokenizer has none by that name.
    func addedTokenID(_ content: String) -> Int32? { addedIDs[content] }

    func encode(_ s: String) -> [Int32] {
        var ids: [Int32] = []
        split(Array(s.utf8), on: rawAdded, into: &ids) { gap, ids in
            let text = String(decoding: gap, as: UTF8.self)
            let normalized = nfc ? text.precomposedStringWithCanonicalMapping : text
            split(Array(normalized.utf8), on: normalizedAdded, into: &ids) { piece, ids in
                encodeOrdinary(String(decoding: piece, as: UTF8.self), into: &ids)
            }
        }
        return ids
    }

    /// Leftmost-longest match of `tokens` in `bytes`; text between matches goes to `gap`.
    private func split(_ bytes: [UInt8], on tokens: [AddedToken], into ids: inout [Int32],
                       gap: (ArraySlice<UInt8>, inout [Int32]) -> Void) {
        var start = 0, i = 0
        scan: while i < bytes.count {
            for token in tokens where bytes.count - i >= token.bytes.count && bytes[i] == token.bytes[0]
                && bytes[i..<(i + token.bytes.count)].elementsEqual(token.bytes) {
                if start < i { gap(bytes[start..<i], &ids) }
                ids.append(token.id)
                i += token.bytes.count
                start = i
                continue scan
            }
            i += 1
        }
        if start < bytes.count { gap(bytes[start...], &ids) }
    }

    private func encodeOrdinary(_ text: String, into ids: inout [Int32]) {
        let ns = text as NSString
        var last = 0
        for match in preTokenizer.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            if match.range.location > last {
                bpe(ns.substring(with: NSRange(location: last, length: match.range.location - last)), into: &ids)
            }
            bpe(ns.substring(with: match.range), into: &ids)
            last = match.range.location + match.range.length
        }
        if last < ns.length { bpe(ns.substring(from: last), into: &ids) }
    }

    private func bpe(_ word: String, into ids: inout [Int32]) {
        var symbols = word.utf8.map { byteIDs[Int($0)] }
        while symbols.count > 1 {
            var best = -1, bestMerge = Merge(rank: .max, id: 0)
            for j in 0..<(symbols.count - 1) {
                if let merge = merges[Self.key(symbols[j], symbols[j + 1])], merge.rank < bestMerge.rank {
                    best = j
                    bestMerge = merge
                }
            }
            if best < 0 { break }
            symbols[best] = bestMerge.id
            symbols.remove(at: best + 1)
        }
        ids.append(contentsOf: symbols)
    }

    private static func key(_ a: Int32, _ b: Int32) -> UInt64 { UInt64(UInt32(a)) << 32 | UInt64(UInt32(b)) }
}

/// GPT-2's reversible byte-to-printable-character map, indexed by byte.
private let byteToUnicode: [Unicode.Scalar] = {
    let printable = Set(Array(33...126) + Array(161...172) + Array(174...255))
    var next = 256
    return (0..<256).map { byte in
        if printable.contains(byte) { return Unicode.Scalar(UInt32(byte))! }
        defer { next += 1 }
        return Unicode.Scalar(UInt32(next))!
    }
}()
