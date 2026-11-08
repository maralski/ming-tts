// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

// The English grammar (frontend/talker_tn/en_tn_*.fst) as the reference applies it to the text
// between entities: cardinals, years, decimals, money with scale suffixes, clock times with am/pm,
// month-day dates, fractions, symbols and a whitelist of abbreviations. Every other word passes
// through unchanged. Readings follow the grammar's outputs on real feed text, except where the
// grammar garbles them (see NormalizeTests).

private let num = #"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"#
private let month = #"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)"#
private let day = #"([12]\d|3[01]|0?[1-9])(?:st|nd|rd|th)?"#
private let year = #"(?:(,?\s+)(\d{4})(?![\w]))?"#

private let moneyNames = ["$": "dollar", "US$": "dollar", "A$": "Australian dollar", "AU$": "Australian dollar",
                          "NZ$": "New Zealand dollar", "C$": "Canadian dollar", "CA$": "Canadian dollar",
                          "HK$": "Hong Kong dollar", "S$": "Singapore dollar", "€": "euro", "£": "pound",
                          "¥": "yuan", "₹": "rupee"]
/// Scale suffixes after an amount, matched without regard to case.
private let moneyScales = ["thousand": "thousand", "million": "million", "billion": "billion", "trillion": "trillion",
                           "k": "thousand", "m": "million", "mn": "million", "b": "billion", "bn": "billion",
                           "t": "trillion", "tn": "trillion"]

private let whitelist = [
    "Mr": "Mister", "Mr.": "Mister", "Mrs": "Misses", "Mrs.": "Misses", "Ms": "Miss", "Ms.": "Miss",
    "Dr": "doctor", "Dr.": "doctor", "Jr": "junior", "Jr.": "junior", "Gov.": "governor", "Gen.": "general",
    "Lt.": "lieutenant", "Capt.": "captain", "Sgt.": "sergeant", "Col.": "colonel", "Maj.": "Major",
    "Rev.": "Reverend", "Hon.": "Honorable", "Mt.": "Mount", "Ltd.": "limited", "Co.": "company", "Ph.D.": "PHD",
    "Mon": "Monday", "Wed": "Wednesday", "Fri": "Friday", "Sat": "Saturday",
    "I'm": "I am", "you're": "you are", "here's": "here is", "Here's": "Here is",
]

private enum GrammarKind {
    case money, scaledNumber, ordinal, clock, monthDay, dayMonth, monthYear, multiplier, fraction, scaled, degrees, numberSign, number
    case symbol, whitelisted, initialism, email, domain, ratio, times
}

private let grammar = RuleScanner<GrammarKind>([
    (.email, #"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"#),
    (.domain, #"(?<![\w@./-])[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|org|net|io|ai|co|gov|edu|au|uk|nz|dev|app|tv)(?:/[\w/-]*)?(?!\w|\.\w)"#),
    (.money, #"(?<![A-Za-z0-9])("# + alternatives(moneyNames.keys) + #")\s?("# + num + #")(?:\s?((?i:"#
        + alternatives(moneyScales.keys) + #"))(?![A-Za-z]))?(?![\d])"#),
    (.clock, #"(?<![\w.:])(\d{1,2})(?:[:.](\d{2}))?"# + meridiem),
    (.monthDay, #"(?<![\w])(?:([Aa]n?)\s+)?"# + month + #"(?:\.\s*|\s+)(?:([Tt]he)\s+)?"# + day + #"(?![\w])"# + year),
    (.dayMonth, #"(?<![\w.])(?:([Tt]he)\s+)?"# + day + #"\s+"# + month + #"(?![\w])"# + year),
    (.monthYear, #"(?<![\w])"# + month + #"\.?\s+(\d{4})(?![\w])"#),
    (.ratio, #"(?<![\w:.])\d+:\d(?![\w:]|\.\d)"#),
    (.times, #"(?<=\d) x (?=\d)"#),
    (.multiplier, #"(?<![\w.])("# + num + #")x(?![\w])"#),
    (.fraction, #"(?<![\w./])(?:(\d+) )?(\d+)/(\d+)(?![\w/]|\.\d)"#),
    (.scaled, #"(?<![\w.])("# + num + #")\s?(bn|tn)(?![\w])"#),
    (.degrees, #"(?<![\w.])(-?(?:"# + num + #"))\s?°(?![CF])"#),
    (.numberSign, #"(?<![\w])(?:[Nn]o\.\s?|#\s?)(?=\d)"#),
    (.scaledNumber, #"(?<![\d.,])("# + num + #")(?=\s+(?:hundred|thousand|million|billion|trillion)\b)"#),
    (.ordinal, #"(?<![\w.])(\d+)(?i:st|nd|rd|th)(?![A-Za-z0-9])"#),
    (.number, #"(?:(?<![\w.,-])(-)|(?<![\d.,-]))(?:(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?|(?<![\w.,-])\.(\d+))(?![\d]|[.,]\d)"#),
    (.whitelisted, #"(?<![\w.'’])(?:"# + alternatives(whitelist.keys) + #")(?![\w'’])"#),
    (.initialism, #"(?<![\w.])(?:[A-Z]\.){2,}(?![\w])"#),
    (.symbol, #"[&+=%|*±√π≠≈]|@(?!\w)|#(?=[A-Za-z])|~(?=\s?[\d$€£])"#),
])

/// `EntityNormalizer.normalize`'s `ordinary`: the grammar over the text between entities.
func ordinaryPart(_ text: NSString, _ range: NSRange) -> String {
    grammar.rewrite(text, in: range, gap: { $0.substring(with: $1) }) { kind, groups, ns, match in
        let end = match.location + match.length
        return render(kind, groups, before: match.location > 0 ? ns.character(at: match.location - 1) : nil,
                      after: end < ns.length ? ns.character(at: end) : nil)
    }
}

private func render(_ kind: GrammarKind, _ g: [String?], before: unichar?, after: unichar?) -> String {
    let value = g[0]!
    switch kind {
    case .money:
        let scale = g[3].flatMap { moneyScales[$0.lowercased()] }
        let amount = g[2]!
        let name = moneyNames[g[1]!]!
        let singular = scale == nil && Double(amount.replacingOccurrences(of: ",", with: "")) == 1
        return padded(spokenNumber(amount) + (scale.map { " " + $0 } ?? "") + " " + (singular ? name : name + "s"),
                      before: before, after: after)
    case .clock:
        return spokenClock(g[1]! + ":" + (g[2] ?? "00"), meridiem: g[3]!.uppercased())
    case .monthDay:
        if g[1] != nil {
            // "an Oct. 18 ambush": attributive, so "an october eighteenth ambush".
            let name = monthName(g[2]!).lowercased()
            let article = (g[1]!.first == "A" ? "A" : "a") + ("aeiou".contains(name.first!) ? "n" : "")
            return article + " " + name + " " + NumberWords.ordinal(String(NumberWords.value(g[4]!)))
                + (g[6].map { (g[5]!.contains(",") ? ", " : " ") + spokenYear(NumberWords.value($0)) } ?? "")
        }
        return spokenDay(article: g[3], month: g[2]!, day: g[4]!, separator: g[5], year: g[6])
    case .dayMonth:
        return spokenDay(article: g[1], month: g[3]!, day: g[2]!, separator: g[4], year: g[5])
    case .monthYear:
        return monthName(g[1]!).lowercased() + " " + spokenYear(NumberWords.value(g[2]!))
    case .multiplier:
        return grammarNumber(String(value.dropLast())) + " times"
    case .fraction:
        let (numerator, denominator) = (g[2]!, g[3]!)
        if g[1] == nil && ["24/7", "9/11"].contains(numerator + "/" + denominator) {
            return NumberWords.cardinal(numerator) + " " + NumberWords.cardinal(denominator)
        }
        return NumberWords.fraction(whole: g[1], numerator: numerator, denominator: denominator)
    case .scaled:
        return grammarNumber(g[1]!) + (g[2] == "bn" ? " billion" : " trillion")
    case .degrees:
        let amount = g[1]!
        return grammarNumber(amount) + (Double(amount) == 1 ? " degree" : " degrees")
    case .numberSign:
        return "number "
    case .scaledNumber:
        return spacedNumber(grammarNumber(value, and: false), before: before, after: after)
    case .ordinal:
        return spacedNumber(NumberWords.ordinal(g[1]!), before: before, after: after)
    case .number:
        return spacedNumber(grammarNumber(value), before: before, after: after)
    case .email, .domain:
        return value.replacingOccurrences(of: ".", with: " dot ").replacingOccurrences(of: "@", with: " at ")
            .replacingOccurrences(of: "/", with: " slash ")
    case .ratio:
        return value.split(separator: ":").map { NumberWords.cardinal(String($0)) }.joined(separator: " to ")
    case .times:
        return " times "
    case .whitelisted:
        return whitelist[value]!
    case .initialism:
        return value.filter { $0 != "." }
    case .symbol:
        let spoken: String
        switch value {
        case "&": spoken = "and"
        case "+": spoken = "plus"
        case "@": spoken = "at"
        case "=": spoken = "equals"
        case "%": spoken = "percent"
        case "~": spoken = "approximately"
        case "±": spoken = "plus or minus"
        case "√": spoken = "square root of"
        case "π": spoken = "pi"
        case "≠": spoken = "not equal to"
        case "≈": spoken = "approximately equals"
        case "|": return ","
        default: return ""
        }
        return padded(spoken, before: before, after: after)
    }
}

/// Words that replace a symbol are spaced off from their neighbours, except brackets and quotes
/// that hold them and punctuation that follows them.
private func padded(_ words: String, before: unichar?, after: unichar?) -> String {
    let opening = before.map { "([{\"'‘“".utf16.contains($0) || isPythonSpace(Unicode.Scalar($0) ?? " ") } ?? true
    let closing = after.map { ",.;:!?)]}\"'’”".utf16.contains($0) || isPythonSpace(Unicode.Scalar($0) ?? " ") } ?? true
    return (opening ? "" : " ") + words + (closing ? "" : " ")
}

/// Number words are spaced off from letters, digits and symbols they were written against ("5G",
/// "2️⃣"), but stay attached to punctuation ("24-year-old").
private func spacedNumber(_ words: String, before: unichar?, after: unichar?) -> String {
    func joins(_ c: unichar?) -> Bool {
        guard let c, let scalar = Unicode.Scalar(c) else { return c.map(UTF16.isLeadSurrogate) ?? false }
        switch scalar.properties.generalCategory {
        case .otherSymbol, .mathSymbol, .modifierSymbol, .currencySymbol, .nonspacingMark, .enclosingMark: return true
        default: return isAlnum(scalar)
        }
    }
    return (joins(before) ? " " : "") + words + (joins(after) ? " " : "")
}

private let plainNumber = Pattern(#"(-)?(\d{1,3}(?:,\d{3})+|\d+)?(?:\.(\d+))?"#)

/// The grammar's reading of a plain number: years for 1000 to 2999, "and" in cardinals, decimals
/// digit by digit, "negative" before integers and "minus" before decimals.
private func grammarNumber(_ text: String, and: Bool = true) -> String {
    let g = plainNumber.fullMatch(text)!
    let whole = g[2]?.replacingOccurrences(of: ",", with: "")
    if let fraction = g[3] {
        return (g[1] == nil ? "" : "minus ") + (whole.map { NumberWords.cardinal($0, and: and) + " " } ?? "") + "point " + NumberWords.digits(fraction)
    }
    let digits = whole!
    let sign = g[1] == nil ? "" : "negative "
    if and, g[2]!.count == 4, case let year = NumberWords.value(digits), (1000...2999).contains(year) { return sign + NumberWords.year(year) }
    return sign + NumberWords.cardinal(digits, and: and)
}

private func monthName(_ abbreviation: String) -> String {
    let months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                  "November", "December"]
    return months.first { $0.hasPrefix(abbreviation.prefix(3)) }!
}

/// "the sixth of november, twenty twenty six", lower case as the grammar writes it.
private func spokenDay(article: String?, month: String, day: String, separator: String?, year: String?) -> String {
    let date = (article ?? "the") + " " + NumberWords.ordinal(String(NumberWords.value(day))) + " of " + monthName(month).lowercased()
    guard let year, let separator else { return date }
    return date + (separator.contains(",") ? ", " : " ") + spokenYear(NumberWords.value(year))
}
