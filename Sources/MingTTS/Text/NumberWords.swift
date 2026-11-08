// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
/// Spoken English numbers in the two styles the reference frontend mixes. Entities
/// (text_rules.py) use `inflect.number_to_words(andword='')`: no "and", a comma between
/// thousands groups. Plain text goes through the English grammar, which says "one hundred
/// and five" and reads 1000 to 2999 as years.
enum NumberWords {
    private static let ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                               "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                               "eighteen", "nineteen"]
    private static let tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    private static let scales = ["", " thousand", " million", " billion", " trillion", " quadrillion", " quintillion",
                                 " sextillion", " septillion", " octillion", " nonillion", " decillion"]

    static func belowHundred(_ n: Int) -> String {
        n < 20 ? ones[n] : tens[n / 10] + (n % 10 == 0 ? "" : " " + ones[n % 10])
    }

    private static func belowThousand(_ n: Int, and: Bool) -> String {
        let hundreds = n / 100, rest = n % 100
        guard hundreds > 0 else { return belowHundred(rest) }
        let head = ones[hundreds] + " hundred"
        return rest == 0 ? head : head + (and ? " and " : " ") + belowHundred(rest)
    }

    /// Three-digit groups, most significant first.
    private static func groups(_ digits: Substring) -> [Int] {
        var result: [Int] = [], end = digits.endIndex
        while end > digits.startIndex {
            let start = digits.index(end, offsetBy: -3, limitedBy: digits.startIndex) ?? digits.startIndex
            result.insert(value(digits[start..<end]), at: 0)
            end = start
        }
        return result
    }

    /// Python's `int()` of a run of digits; like `\d`, it accepts any Unicode decimal digit.
    static func value(_ digits: some StringProtocol) -> Int {
        digits.reduce(0) { $0 * 10 + ($1.wholeNumberValue ?? 0) }
    }

    static func digits(_ s: some StringProtocol) -> String {
        s.compactMap { $0.wholeNumberValue }.map { ones[$0] }.joined(separator: " ")
    }

    /// `inflect.number_to_words(text, andword='')` with hyphens spoken as spaces; nil past decillions.
    static func inflect(_ text: String) -> String? {
        let parts = text.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
        let whole = parts[0].drop { $0 == "0" }
        var words: String
        if parts[0].isEmpty {
            words = ""
        } else if whole.isEmpty {
            words = "zero"
        } else {
            let groups = groups(whole)
            guard groups.count <= scales.count else { return nil }
            // inflect separates groups with commas, except before a final group of one word.
            words = ""
            for (i, g) in groups.enumerated() where g > 0 {
                let last = i == groups.count - 1
                if !words.isEmpty { words += last && g < 100 ? " " : ", " }
                words += belowThousand(g, and: false) + scales[groups.count - 1 - i]
            }
        }
        if parts.count > 1 {
            words += (words.isEmpty ? "" : " ") + "point" + (parts[1].isEmpty ? "" : " " + digits(parts[1]))
        }
        return words
    }

    /// The grammar's cardinal: "and" only inside the final group, or before a final group under
    /// a hundred that follows thousands. Digit strings with a leading zero are read digit by digit.
    static func cardinal(_ digitString: String, and: Bool = true) -> String {
        let s = Substring(digitString)
        if s.count > 1 && s.first == "0" { return digits(s) }
        let groups = groups(s)
        guard groups.count <= scales.count else { return digits(s) }
        if groups.allSatisfy({ $0 == 0 }) { return "zero" }
        var chunks: [String] = []
        for (i, g) in groups.enumerated() where g > 0 {
            let last = i == groups.count - 1
            var chunk = belowThousand(g, and: and && last) + scales[groups.count - 1 - i]
            if and && last && g < 100 && groups.count > 1 && groups[i - 1] > 0 { chunk = "and " + chunk }
            chunks.append(chunk)
        }
        return chunks.joined(separator: " ")
    }

    /// Years as the grammar reads 1000 through 2999.
    static func year(_ n: Int) -> String {
        let high = n / 100, low = n % 100
        if high % 10 == 0 && low < 10 { return cardinal(String(n), and: false) }
        if low == 0 { return belowHundred(high) + " hundred" }
        return belowHundred(high) + (low < 10 ? " oh " : " ") + belowHundred(low)
    }

    static func ordinal(_ n: String) -> String {
        ordinalize(cardinal(n, and: false))
    }

    static func ordinalize(_ words: String) -> String {
        var parts = words.split(separator: " ").map(String.init)
        guard let last = parts.popLast() else { return words }
        let irregular = ["one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                         "nine": "ninth", "twelve": "twelfth"]
        let spoken = irregular[last] ?? (last.hasSuffix("y") ? last.dropLast() + "ieth" : last + "th")
        return (parts + [spoken]).joined(separator: " ")
    }

    /// The grammar's fractions: "one half", "three quarters", "five eighths", "three and a half".
    static func fraction(whole: String?, numerator: String, denominator: String) -> String {
        let n = value(numerator), d = value(denominator)
        let unit: String
        switch d {
        case 2: unit = n == 1 ? "half" : "halves"
        case 4: unit = n == 1 ? "quarter" : "quarters"
        default: unit = ordinal(denominator) + (n == 1 ? "" : "s")
        }
        let top = cardinal(numerator, and: false)
        guard let whole else { return top + " " + unit }
        let article = n == 1 && (d == 2 || d == 4) ? "a" : top
        return cardinal(whole, and: false) + " and " + article + " " + unit
    }
}
