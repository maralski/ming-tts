// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The fragments `MingTTS.synthesize` generates one at a time for `request`: `prepareText`, then
/// `splitText`. Throws where the reference's `split_text` raises.
func prepareFragments(_ request: MingRequest, maxLength: Int) throws -> [String] {
    let prepared = prepareText(request.text, task: request.task, language: request.language)
    let fragments = splitText(prepared, maxLength: maxLength)
    guard !fragments.isEmpty else { throw MingTTSError.badInput("text produced no synthesizable fragments: \(request.text)") }
    return fragments
}

/// `frontend.text_processing.prepare_text`: Markdown cleanup that keeps paralinguistic tags,
/// normalization, then a closing period. Japanese, Korean, German, French, Spanish and dialect
/// text skips the last two. Chinese normalization is not ported, so English normalization runs
/// only for English, or for no language when the text holds no Han character (the reference
/// detects Chinese that way). `normalizing: false` is the reference's `text_normalization=False`.
func prepareText(_ text: String, task: MingTask, language: MingLanguage?, normalizing: Bool = true) -> String {
    let cleaned = pythonStrip(stripMarkdown(text))
    if case .dialect = task { return cleaned }
    if language?.preservesText == true { return cleaned }
    let english = language == .english || (language == nil && !cleaned.unicodeScalars.contains(where: isHan))
    return addPeriod(normalizing && english ? normalizeEnglish(cleaned) : cleaned)
}

/// TalkerTN's `re_contains_chinese`.
private func isHan(_ c: Unicode.Scalar) -> Bool {
    (0x3400...0x9FFF).contains(c.value) || (0x20000...0x3134F).contains(c.value)
}

/// `split_text`: whole clauses grouped up to `maxLength`. Returns no fragments where the reference raises.
func splitText(_ prepared: String, maxLength: Int) -> [String] {
    precondition(maxLength >= 1, "maxLength must be positive")
    let fragments = cutBySemanticLength(prepared, maxLength: maxLength).filter { !pythonStrip($0).isEmpty }
    guard fragments.contains(where: { $0.unicodeScalars.contains(where: isAlnum) }) else { return [] }
    return fragments
}

// MARK: - frontend/toolkit.py

let paralinguisticTags: Set<String> = [
    "[Breathing]", "[Cough]", "[Sigh]", "[Laughter]", "[Surprise-ah]", "[Surprise-oh]",
    "[Dissatisfaction-hnn]", "[Uhm]", "[Confirmation-en]", "[Question-ei]", "[Question-ah]",
    "[Question-en]", "[Question-oh]", "[Question-huh]",
]

/// `preprosess_text`. Python's `.` excludes only `\n`, so patterns spell it `[^\n]`.
func stripMarkdown(_ text: String) -> String {
    var text = replace(#"```[^\n`]*\n([\s\S]*?)```"#, in: text, with: "$1")
    text = replace(#"``([^\n]+?)``"#, in: text, with: "$1")
    text = replace(#"`([^\n]+?)`"#, in: text, with: "$1")
    text = text.replacingOccurrences(of: "`", with: "")
    text = unwrapMarkdownLinks(text)
    text = replace(#"(?:^|(?<=\n))[\s\x1c-\x1f]{0,3}#{1,6}[ \t]+"#, in: text, with: "")
    return replace(#"\*\*([\s\S]+?)\*\*"#, in: text, with: "$1")
}

/// Replaces `[label](url)` and `![label](url)` with `label`, balancing parentheses in the url,
/// unless `[label]` is a paralinguistic tag.
private func unwrapMarkdownLinks(_ text: String) -> String {
    let s = Array(text.unicodeScalars)
    var out = String.UnicodeScalarView()
    var offset = 0, pos = 0
    while pos < s.count {
        // finditer for !?\[([^]\n]*)\]\(
        guard let (start, labelStart, labelEnd, matchEnd) = nextLinkOpening(s, from: pos) else { break }
        pos = matchEnd
        if start < offset { continue }
        var depth = 1, end = matchEnd
        while end < s.count && depth > 0 && s[end] != "\n" {
            if s[end] == "\\" { end += 2; continue }
            if s[end] == "(" { depth += 1 } else if s[end] == ")" { depth -= 1 }
            end += 1
        }
        if depth > 0 { continue }
        out.append(contentsOf: s[offset..<start])
        let label = s[labelStart..<labelEnd]
        let tag = "[" + String(String.UnicodeScalarView(label)) + "]"
        out.append(contentsOf: paralinguisticTags.contains(tag) ? s[start..<min(end, s.count)] : label)
        offset = min(end, s.count)
    }
    out.append(contentsOf: s[min(offset, s.count)...])
    return String(out)
}

private func nextLinkOpening(_ s: [Unicode.Scalar], from: Int) -> (Int, Int, Int, Int)? {
    var i = from
    while i < s.count {
        let open = s[i] == "!" && i + 1 < s.count && s[i + 1] == "[" ? i + 1 : (s[i] == "[" ? i : -1)
        if open >= 0 {
            var j = open + 1
            while j < s.count && s[j] != "]" && s[j] != "\n" { j += 1 }
            if j + 1 < s.count && s[j] == "]" && s[j + 1] == "(" { return (i, open + 1, j, j + 2) }
        }
        i += 1
    }
    return nil
}

/// `add_period`: end with sentence punctuation, promoting a trailing comma or semicolon.
func addPeriod(_ text: String) -> String {
    var s = Array(text.unicodeScalars)
    while let last = s.last, isPythonSpace(last) { s.removeLast() }
    guard let last = s.last else { return "." }
    let hasHan = s.contains { (0x4E00...0x9FFF).contains($0.value) }
    if "，,、;；".unicodeScalars.contains(last) {
        s[s.count - 1] = hasHan ? "。" : "."
    } else if isAlnum(last) {
        let isHan = (0x4E00...0x9FFF).contains(last.value) || (0x3400...0x4DBF).contains(last.value)
        s.append(isHan ? "。" : ".")
    }
    return String(String.UnicodeScalarView(s))
}

// MARK: - frontend/text_segment_cut.py

private let sentenceEnds = Set("。！？.!?".unicodeScalars)
private let clauseEnds = Set("，,;；".unicodeScalars)
private let asciiPunctuationAndSpace = Set(##"!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ "##.unicodeScalars).union("\t\n\r\u{0B}\u{0C}".unicodeScalars)

/// `cut_text_by_semantic_length(...)["fragments"]` with `min_tail_length` 5.
func cutBySemanticLength(_ text: String, maxLength: Int, minTail: Int = 5) -> [String] {
    guard hasValidContent(text) else { return [] }

    // Shield sentence punctuation inside URLs, emails, numbers, initialisms and titles.
    let punct = Array(".,;!?，。！？".unicodeScalars)
    let present = Set(text.unicodeScalars)
    var shields: [Unicode.Scalar: Unicode.Scalar] = [:], restore: [Unicode.Scalar: Unicode.Scalar] = [:]
    var code: UInt32 = 0xE000
    for p in punct {
        while present.contains(Unicode.Scalar(code)!) { code += 1 }
        shields[p] = Unicode.Scalar(code)!
        restore[Unicode.Scalar(code)!] = p
        code += 1
    }
    let processed = replace(structures, in: text) { token in
        var token = Array(token.unicodeScalars), tail: [Unicode.Scalar] = []
        let lower = String(String.UnicodeScalarView(token.prefix(8))).lowercased()
        if lower.hasPrefix("http://") || lower.hasPrefix("https://") || lower.hasPrefix("www.") {
            while let last = token.last, ".,;!?".unicodeScalars.contains(last) { tail.insert(token.removeLast(), at: 0) }
        }
        return String(String.UnicodeScalarView(token.map { shields[$0] ?? $0 } + tail))
    }

    if semanticLength(processed) <= maxLength { return [text] }

    var sentences: [String] = []
    var current = String.UnicodeScalarView()
    for c in processed.unicodeScalars {
        current.append(c)
        if sentenceEnds.contains(c) {
            let sentence = pythonStrip(String(current))
            if !sentence.isEmpty { sentences.append(sentence) }
            current = String.UnicodeScalarView()
        }
    }
    if !current.isEmpty { sentences.append(pythonStrip(String(current))) }

    var fragments: [String] = []
    for sentence in sentences {
        var clauses: [String] = []
        var current = String.UnicodeScalarView()
        func close() {
            let clause = pythonStrip(String(current))
            if !clause.isEmpty && hasValidContent(clause) { clauses.append(clause) }
            else if !clause.isEmpty && !clauses.isEmpty { clauses[clauses.count - 1] += clause }
            current = String.UnicodeScalarView()
        }
        for c in sentence.unicodeScalars {
            current.append(c)
            if clauseEnds.contains(c) { close() }
        }
        if !current.isEmpty { close() }

        var i = 0
        while i < clauses.count {
            let clause = clauses[i]
            let length = semanticLength(clause)
            if length < minTail && i + 1 < clauses.count {
                let combined = clause + clauses[i + 1]
                if semanticLength(combined) <= maxLength {
                    appendFragment(&fragments, combined, maxLength: maxLength, minTail: minTail)
                    i += 2
                    continue
                }
            }
            if length > maxLength {
                for part in splitLongFragment(clause, maxLength: maxLength) {
                    appendFragment(&fragments, part, maxLength: maxLength, minTail: minTail)
                }
            } else {
                appendFragment(&fragments, clause, maxLength: maxLength, minTail: minTail)
            }
            i += 1
        }
    }
    return fragments.map { String(String.UnicodeScalarView($0.unicodeScalars.map { restore[$0] ?? $0 })) }
}

private let latinLetters = #"A-Za-z\x{c0}-\x{d6}\x{d8}-\x{f6}\x{f8}-\x{24f}\x{1e00}-\x{1eff}"#
private let structures = try! NSRegularExpression(pattern:
    #"(?i:https?://|www\.)[^\s\x1c-\x1f，。！？]+|"#
    + #"[\p{L}\p{N}_.+-]+@[\p{L}\p{N}_.-]+\.[A-Za-z]{2,}|"#
    + #"\d+(?:[.,]\d+)+|(?:[A-Za-z]\.){2,}|"#
    + #"(?i:(?<![A-Za-z])(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.)"#)

/// `get_semantic_length`: one per CJK character or kana/hangul syllable, one per Latin word.
func semanticLength(_ text: String) -> Int {
    let s = Array(text.precomposedStringWithCanonicalMapping.unicodeScalars)
    func isLatin(_ c: Unicode.Scalar) -> Bool {
        switch c.value {
        case 0x41...0x5A, 0x61...0x7A, 0xC0...0xD6, 0xD8...0xF6, 0xF8...0x24F, 0x1E00...0x1EFF: true
        default: false
        }
    }
    func isCJK(_ c: Unicode.Scalar) -> Bool {
        switch c.value {
        case 0x4E00...0x9FFF, 0x3041...0x3096, 0x309D...0x309F, 0x30A1...0x30FA, 0x30FC...0x30FF, 0xAC00...0xD7A3: true
        default: false
        }
    }
    var count = 0, i = 0
    while i < s.count {
        if isCJK(s[i]) { count += 1; i += 1; continue }
        guard isLatin(s[i]) else { i += 1; continue }
        count += 1
        while i < s.count && isLatin(s[i]) { i += 1 }
        while i + 1 < s.count && (s[i] == "'" || s[i] == "’") && isLatin(s[i + 1]) {
            i += 1
            while i < s.count && isLatin(s[i]) { i += 1 }
        }
    }
    return count
}

private func hasValidContent(_ text: String) -> Bool {
    text.unicodeScalars.contains { !asciiPunctuationAndSpace.contains($0) }
}

private func appendFragment(_ fragments: inout [String], _ newText: String, maxLength: Int, minTail: Int) {
    let leading = Set("，,:; \t\n\r\u{0B}\u{0C}".unicodeScalars)
    let newText = String(String.UnicodeScalarView(newText.unicodeScalars.drop(while: leading.contains)))
    guard hasValidContent(newText) else { return }
    guard let last = fragments.last else { fragments.append(newText); return }
    let newLength = semanticLength(newText)
    guard semanticLength(last) + newLength <= maxLength else { fragments.append(newText); return }
    if let end = last.unicodeScalars.last, "。！？".unicodeScalars.contains(end), newLength < minTail {
        fragments.append(newText)
        return
    }
    let startsAlnum = newText.unicodeScalars.first.map { $0.isASCII && ($0.properties.isAlphabetic || ("0"..."9").contains($0)) } ?? false
    let separator = last.unicodeScalars.last != " " && startsAlnum ? " " : ""
    fragments[fragments.count - 1] = last + separator + newText
}

private let clausePattern = try! NSRegularExpression(pattern: #"[，、；。！？,;.!?]|[\s\x1c-\x1f]+[-•·]"#)

/// `split_long_fragment`: group punctuation-bounded clauses up to `maxLength`; never cut inside one.
private func splitLongFragment(_ text: String, maxLength: Int) -> [String] {
    if semanticLength(text) <= maxLength { return [text] }
    let ns = text as NSString
    var clauses: [String] = []
    var last = 0
    for match in clausePattern.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
        if match.range.location > last { clauses.append(ns.substring(with: NSRange(location: last, length: match.range.location - last))) }
        let separator = ns.substring(with: match.range)
        if clauses.isEmpty { clauses.append(separator) } else { clauses[clauses.count - 1] += separator }
        last = match.range.location + match.range.length
    }
    if last < ns.length { clauses.append(ns.substring(from: last)) }

    var fragments: [String] = []
    var current = ""
    for clause in clauses {
        let length = semanticLength(clause)
        if length > maxLength {
            if !current.isEmpty { fragments.append(current); current = "" }
            fragments.append(clause)
        } else if semanticLength(current) + length <= maxLength {
            current += clause
        } else {
            if !current.isEmpty { fragments.append(current) }
            current = clause
        }
    }
    if !current.isEmpty { fragments.append(current) }
    return fragments
}

// MARK: - Python string semantics

/// Python `str.isspace`: Unicode White_Space plus the ASCII separators U+001C to U+001F.
func isPythonSpace(_ c: Unicode.Scalar) -> Bool {
    c.properties.isWhitespace || (0x1C...0x1F).contains(c.value)
}

/// Python `str.isalnum` for one code point.
func isAlnum(_ c: Unicode.Scalar) -> Bool {
    switch c.properties.generalCategory {
    case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter,
         .decimalNumber, .letterNumber, .otherNumber: true
    default: false
    }
}

/// Python `str.strip()`.
func pythonStrip(_ text: String) -> String {
    let s = text.unicodeScalars
    guard let first = s.firstIndex(where: { !isPythonSpace($0) }), let last = s.lastIndex(where: { !isPythonSpace($0) })
    else { return "" }
    return String(s[first...last])
}

private func replace(_ pattern: String, in text: String, with template: String) -> String {
    let regex = try! NSRegularExpression(pattern: pattern)
    return regex.stringByReplacingMatches(in: text, range: NSRange(location: 0, length: (text as NSString).length), withTemplate: template)
}

private func replace(_ regex: NSRegularExpression, in text: String, with transform: (String) -> String) -> String {
    let ns = text as NSString
    var out = ""
    var last = 0
    for match in regex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
        out += ns.substring(with: NSRange(location: last, length: match.range.location - last))
        out += transform(ns.substring(with: match.range))
        last = match.range.location + match.range.length
    }
    return out + ns.substring(from: last)
}
