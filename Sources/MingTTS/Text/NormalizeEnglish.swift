// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// English text normalization, the reference `TalkerTN` path that `prepare_text` runs for English
/// before splitting: whole entities first (frontend/talker_tn/text_rules.py), the English grammar
/// (en_tn FST) on the text between them, then `normalize_text_kit`. Numbers, money, percentages,
/// dates, times, ordinals, units and common abbreviations become words. Paralinguistic tags pass
/// through untouched.
///
/// Where the reference grammar mangles a reading ("1990s" as seconds, "1 in 5" as inches, "3:30pm"
/// as "thirtypm", "#AI" as "number AI") this gives the reading the rules intend; NormalizeTests
/// lists every such case beside the reference output.
func normalizeEnglish(_ text: String) -> String {
    let text = preprocess(text)
    let ns = text as NSString
    var out = "", offset = 0
    func normalized(_ segment: String) -> String {
        guard !pythonStrip(segment).isEmpty else { return segment }
        let lead = segment.unicodeScalars.first.map(isPythonSpace) == true ? " " : ""
        let trail = segment.unicodeScalars.last.map(isPythonSpace) == true ? " " : ""
        return lead + pythonStrip(normalizeEntities(segment)) + trail
    }
    for match in paralinguisticTagRegex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
        out += normalized(ns.substring(with: NSRange(location: offset, length: match.range.location - offset)))
        out += ns.substring(with: match.range)
        offset = match.range.location + match.range.length
    }
    out += normalized(ns.substring(from: offset))
    return normalizeTextKit(out)
}

/// `normalize_text_kit`: drop controls and other unprintables, collapse whitespace, strip.
func normalizeTextKit(_ text: String) -> String {
    var out = String.UnicodeScalarView(), pendingSpace = false
    for c in text.precomposedStringWithCanonicalMapping.unicodeScalars {
        if isPythonSpace(c) || c.properties.generalCategory == .spaceSeparator {
            pendingSpace = !out.isEmpty
            continue
        }
        switch c.properties.generalCategory {
        case .control, .surrogate, .privateUse, .unassigned, .lineSeparator, .paragraphSeparator: continue
        case .format where c.value != 0x200C && c.value != 0x200D: continue
        default: break
        }
        if pendingSpace { out.append(" ") }
        pendingSpace = false
        out.append(c)
    }
    return String(out)
}

/// `TalkerPreProcessor` without its Chinese space removal, which glues English words to symbols
/// ("weeks, ~80" became "weeks,~80").
private func preprocess(_ text: String) -> String {
    var scalars = String.UnicodeScalarView()
    for c in text.precomposedStringWithCanonicalMapping.unicodeScalars {
        switch c.value {
        case 0xFF01...0xFF5E: scalars.append(Unicode.Scalar(c.value - 0xFEE0)!)
        case 0x3000: scalars.append(" ")
        case 0x2212: scalars.append("-")
        default: scalars.append(c)
        }
    }
    return stripMarkdown(String(scalars)).split(separator: "\n", omittingEmptySubsequences: false)
        .map { normalizeTextKit(String($0)) }.joined(separator: "\n")
}

private let paralinguisticTagRegex = try! NSRegularExpression(
    pattern: paralinguisticTags.sorted().map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|"))

// MARK: - Regex plumbing

/// A pattern compiled for searching and, separately, for Python's `fullmatch`.
struct Pattern: Sendable {
    let source: String
    let search: NSRegularExpression
    let full: NSRegularExpression

    init(_ source: String) {
        self.source = source
        search = try! NSRegularExpression(pattern: source)
        full = try! NSRegularExpression(pattern: #"\A(?:"# + source + #")\z"#)
    }

    /// Capture groups of a full match, nil for groups that did not take part.
    func fullMatch(_ text: String) -> [String?]? {
        let ns = text as NSString
        guard let m = full.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) else { return nil }
        return (0..<m.numberOfRanges).map { m.range(at: $0).location == NSNotFound ? nil : ns.substring(with: m.range(at: $0)) }
    }
}

/// Python's `re.sub` over named alternatives: one leftmost scan, the first listed rule winning at a
/// position, each match rendered by its rule.
struct RuleScanner<Kind: Sendable>: Sendable {
    private let regex: NSRegularExpression
    private let rules: [(kind: Kind, group: Int, captures: Int)]

    init(_ rules: [(Kind, String)]) {
        var group = 1, located: [(Kind, Int, Int)] = []
        for (kind, pattern) in rules {
            let captures = (try! NSRegularExpression(pattern: pattern)).numberOfCaptureGroups
            located.append((kind, group, captures))
            group += 1 + captures
        }
        regex = try! NSRegularExpression(pattern: rules.map { "(" + $0.1 + ")" }.joined(separator: "|"))
        self.rules = located
    }

    /// Replaces each match in `range` with `render(kind, groups, text, range)`, where `groups[0]` is
    /// the match and the rest are the rule's own captures; the text between matches goes through
    /// `gap`. Lookarounds see past `range`, so a rule can tell what surrounds it.
    func rewrite(_ ns: NSString, in range: NSRange? = nil, gap: (NSString, NSRange) -> String,
                 render: (Kind, [String?], NSString, NSRange) -> String) -> String {
        let range = range ?? NSRange(location: 0, length: ns.length)
        var out = "", offset = range.location
        for m in regex.matches(in: ns as String, options: .withTransparentBounds, range: range) {
            guard let rule = rules.first(where: { m.range(at: $0.group).location != NSNotFound }) else { continue }
            let groups = (rule.group...(rule.group + rule.captures)).map { i in
                m.range(at: i).location == NSNotFound ? nil : ns.substring(with: m.range(at: i))
            }
            out += gap(ns, NSRange(location: offset, length: m.range.location - offset))
            out += render(rule.kind, groups, ns, m.range)
            offset = m.range.location + m.range.length
        }
        return out + gap(ns, NSRange(location: offset, length: range.location + range.length - offset))
    }
}

func alternatives(_ values: some Sequence<String>) -> String {
    values.sorted { $0.count != $1.count ? $0.count > $1.count : $0 < $1 }
        .map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|")
}
