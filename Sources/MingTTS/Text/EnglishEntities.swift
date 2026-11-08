// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

// The English half of frontend/talker_tn/text_rules.py `EntityNormalizer`: each entity is read
// once, as a whole, and the text between entities goes to `normalizeOrdinary`.

private let decimal = #"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"#
private let number = #"[+-]?(?:\d+\s+\d+/\d+|\d+/\d+|(?:"# + decimal + #")(?:[eE][+-]?\d+)?)"#
private let base = "(?:" + number + "|[A-Za-z])"
private let operand = #"(?:√\s*)?"# + base + #"(?:\^[+-]?\d+|[²³])?"#
private let mathOperator = "(?:>=|<=|[+*×÷=<>≠≤≥≈-])"
private let expression = operand + #"(?:\s*"# + mathOperator + #"\s*"# + operand + ")*"
private let rangeDash = #"\s*[-–—~～至到]\s*"#
private let quantity = number + "(?:" + rangeDash + number + ")?"
private let left = "(?<![A-Za-z0-9_.])"
private let right = #"(?![A-Za-z0-9_]|\.\d)"#
private let datePattern = #"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}/\d{1,2}/\d{4})"#
private let timePattern = #"\d{1,2}:\d{2}(?::\d{2})?"#
/// am/pm after a time. A final dot that ends the sentence stays behind as punctuation.
let meridiem = #"\s?([AaPp])\.?[Mm](?:\.(?!\s*$|\s+[A-Z]))?(?![A-Za-z])"#

private let units: [String: String] = {
    let base = ["m": "meter", "g": "gram", "s": "second", "W": "watt", "V": "volt", "A": "ampere", "Hz": "hertz",
                "Pa": "pascal", "J": "joule", "L": "liter", "l": "liter", "mol": "mole", "Ω": "ohm", "N": "newton",
                "F": "farad", "B": "byte", "b": "bit"]
    let prefixes = ["": "", "k": "kilo", "M": "mega", "G": "giga", "T": "tera", "d": "deci", "c": "centi",
                    "m": "milli", "μ": "micro", "µ": "micro", "u": "micro", "n": "nano", "p": "pico", "f": "femto"]
    var result: [String: String] = [:]
    for (suffix, name) in base { for (prefix, prefixName) in prefixes { result[prefix + suffix] = prefixName + name } }
    result.merge([
        "h": "hour", "hr": "hour", "min": "minute", "sec": "second", "day": "day", "days": "day", "ft": "foot",
        "in": "inch", "lb": "pound", "lbs": "pound", "mi": "mile", "oz": "ounce", "°C": "degree Celsius",
        "℃": "degree Celsius", "°F": "degree Fahrenheit", "℉": "degree Fahrenheit", "bit": "bit", "byte": "byte",
        "KB": "kilobyte", "KHz": "kilohertz", "kWh": "kilowatt hour", "Wh": "watt hour", "mAh": "milliampere hour",
        "Ah": "ampere hour", "cal": "calorie", "kcal": "kilocalorie", "mmHg": "millimeter of mercury",
        "bpm": "beat per minute", "rpm": "revolution per minute", "ha": "hectare", "t": "ton", "dB": "decibel",
        "rad": "radian", "㎡": "square meter", "㎥": "cubic meter",
    ]) { $1 }
    for key in ["m", "dm", "cm", "mm", "km", "ft", "in", "mi"] {
        let name = result[key]!
        for (power, word) in [("2", "square"), ("3", "cubic")] {
            for notation in [power, "^" + power, power == "2" ? "²" : "³"] { result[key + notation] = word + " " + name }
        }
    }
    return result
}()

/// The English names of `currency_en`, keyed by every symbol the reference's money patterns accept.
private let currencies: [String: String] = [
    "$": "dollar", "US$": "dollar", "USD": "dollar", "€": "euro", "EUR": "euro", "£": "pound", "￡": "pound",
    "GBP": "pound", "¥": "yuan", "￥": "yuan", "CNY": "yuan", "RMB": "yuan", "HK$": "Hong Kong dollar",
    "HKD": "Hong Kong dollar", "HK＄": "Hong Kong dollar", "A$": "Australian dollar", "AUD": "Australian dollar",
    "CAD$": "Canadian dollar", "CAD": "Canadian dollar", "JPY": "yen", "J￥": "yen", "JP￥": "yen", "JPY￥": "yen",
    "JP¥": "yen", "INR": "rupee", "₹": "rupee", "RUB": "ruble", "₽": "ruble", "CHF": "Swiss franc",
    "SGD": "Singapore dollar", "KRW": "won", "R$": "Brazilian real", "THB": "baht", "฿": "baht",
    "AED": "UAE dirham", "SAR": "Saudi riyal", "SEK": "Swedish krona", "ARS": "Argentine peso", "TRY": "Turkish lira",
]

private let scaleWords = ["百": "hundred", "千": "thousand", "万": "ten thousand", "亿": "hundred million",
                          "兆": "trillion", "thousand": "thousand", "million": "million", "billion": "billion",
                          "trillion": "trillion"]

/// "in" counts as inches only when written against the number: "52-48 in the Senate" is not a measure.
private let unit = "(?:" + units.keys.sorted { $0.count != $1.count ? $0.count > $1.count : $0 < $1 }
    .map { $0 == "in" ? #"(?<=\d)in"# : NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "|") + ")"
private let currency = "(?:" + alternatives(currencies.keys) + ")"
private let scale = "(?:" + alternatives(scaleWords.keys) + ")"

private let moneyBefore = ("([+-]?" + currency + #")\s*("# + quantity + #")(?:\s*("# + scale + #"))?(?![A-Za-z0-9_]|\.\d)"#)
private let moneyAfter = ("(" + quantity + #")\s*("# + scale + #")?\s*("# + currency + ")(?![A-Za-z])")
private let moneyRange = moneyBefore + rangeDash + moneyBefore
private let measure = ("(" + quantity + #")\s*("# + unit + #")(?:\s*/\s*("# + unit + ")|(ps))?(?![A-Za-z0-9_²³^])")
private let serialLabel = "(?:" + alternatives(["id", "ID", "身份证", "手机", "手机号", "电话", "尾号", "学号"] +
    ["身份证号", "身份证号码", "手机号码", "电话号码", "号码", "邮编", "账号", "编号", "订单号", "序列号"]) +
    "|(?i:phone(?: number)?|telephone(?: number)?|mobile(?: number)?|ZIP(?: code)?|postal(?: code)?|postcode|call|contact|tel|fax))"
private let serial = ("(" + serialLabel + #"\s*(?:(?i:is)|是|为)?\s*[:：#]?\s*)(\+?[\d(][\d ()-]*\d)"# + right)
private let dateTime = Pattern("(" + datePattern + ")[T ](" + timePattern + #")(Z|[+-]\d{2}:\d{2})?"#)
private let numberPattern = Pattern(number)
private let quantityPair = Pattern("(" + number + ")" + rangeDash + "(" + number + ")")
private let operandPattern = Pattern(operand)
private let powerPattern = Pattern("(" + base + #")(?:\^([+-]?\d+)|([²³]))"#)
private let operatorStep = try! NSRegularExpression(pattern: #"\A\s*("# + mathOperator + #")\s*("# + operand + ")")

enum EntityKind {
    case url, formula, dateTime, date, serial, time, decade, moneyRange, moneyBefore, moneyAfter, measure, ratio
    case scientific, power, version, math, range, programming, ordinal, unicodeNumber, technical, identifier, longNumber
}

private let entities = RuleScanner<EntityKind>([
    (.url, #"(?i:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+%=,;]*[A-Za-z0-9\-_~/#\[\]@$&'()*+%=]"#),
    (.formula, #"\$\$\s*"# + expression + #"\s*\$\$|\$\s*"# + expression + #"\s*\$"#),
    (.dateTime, left + dateTime.source + right),
    (.date, left + datePattern + right),
    (.serial, "(?<![A-Za-z0-9_])" + serial),
    // The reference leaves am/pm to the grammar, which then glues it on: "three thirtypm".
    (.time, left + timePattern + "(?:" + rangeDash + timePattern + #")?(?![\d:])(?:"# + meridiem + ")?"),
    // Without this the measure rule reads "1990s" as 1990 seconds.
    (.decade, left + #"'?(?:\d{2})?\d0s(?![A-Za-z0-9_])"#),
    (.moneyRange, left + moneyRange),
    (.moneyBefore, left + moneyBefore),
    (.moneyAfter, left + moneyAfter),
    // Not after a currency symbol: "$100m" is money, which the grammar reads.
    (.measure, left + "(?<![$€£¥₹])" + measure),
    (.ratio, left + quantity + #"\s*[%‰‱]"#),
    (.scientific, left + "[+-]?(?:" + decimal + #")[eE][+-]?\d+"# + right),
    (.power, left + "(?:" + number + #"|[A-Za-z])(?:\^[+-]?\d+|[²³])"# + right),
    (.version, left + #"(?:[vV]\d+(?:\.\d+)+|\d+(?:\.\d+){2,})"# + right),
    (.math, left + operand + #"(?:\s*"# + mathOperator + #"\s*"# + operand + ")+" + right),
    (.range, left + number + rangeDash + number + right),
    (.programming, "(?<![A-Za-z0-9_])(?:[Cc]\\+\\+|[Cc]#)(?![A-Za-z0-9_])"),
    (.ordinal, left + #"\d+(?:st|nd|rd|th)"# + right),
    (.unicodeNumber, "[Ⅰ-Ⅻⅰ-ⅻ①-⑳㉑-㉟㊱-㊿⓪❶-❿⓫-⓴⓵-⓾]"),
    (.technical, "(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+(?![A-Za-z0-9_])"),
    (.identifier, "(?<![A-Za-z0-9_])[A-Za-z]+\\d+[A-Za-z0-9]*(?![A-Za-z0-9_])"),
    (.longNumber, left + #"(?:\+\d{1,3}[ -]+)?[+-]?\d{5,}"# + right),
])

/// `EntityNormalizer.normalize(text, chinese=False)`.
func normalizeEntities(_ text: String) -> String {
    entities.rewrite(text as NSString, gap: ordinaryPart) { kind, g, ns, range in
        let value = g[0]!
        let before = { (n: Int) in ns.substring(with: NSRange(location: max(0, range.location - n), length: min(n, range.location))) }
        switch kind {
        case .formula:
            return math(pythonStrip(value.trimmingCharacters(in: ["$"])), formula: true)
        case .moneyRange:
            return moneyValue(g[1]!, g[2]!, g[3], original: value) + " to " + moneyValue(g[4]!, g[5]!, g[6], original: value)
        case .moneyBefore:
            return moneyValue(g[1]!, g[2]!, g[3], original: value)
        case .moneyAfter:
            return moneyValue(g[3]!, g[1]!, g[2], original: value)
        case .measure:
            let denominator = g[4] != nil ? "s" : g[3]
            return spokenQuantity(g[1]!) + " " + plural(units[g[2]!]!, g[1]!) + (denominator.map { " per " + units[$0]! } ?? "")
        case .ratio:
            let symbol = String(value.last!)
            let name = ["%": "percent", "‰": "per thousand", "‱": "per ten thousand"][symbol]!
            return spokenQuantity(pythonStrip(String(value.dropLast()))) + " " + name
        case .date:
            return spokenDate(value)
        case .dateTime:
            return spokenDateTime(value)
        case .serial:
            let digits = g[2]!.compactMap { c -> String? in
                c == "+" ? "plus" : c.wholeNumberValue.map { NumberWords.digits(String($0)) }
            }
            let label = g[1]! as NSString
            return pythonStrip(ordinaryPart(label, NSRange(location: 0, length: label.length))) + " " + digits.joined(separator: " ")
        case .time:
            return spokenTimes(value)
        case .decade:
            return spokenDecade(value)
        case .scientific, .range:
            return spokenQuantity(value)
        case .math:
            return math(value, formula: false)
        case .power:
            return spokenOperand(value)
        case .version:
            return spokenVersion(value, context: before(20))
        case .programming:
            return value.uppercased() == "C++" ? "C plus plus" : "C sharp"
        case .ordinal:
            return NumberWords.ordinal(String(value.dropLast(2)))
        case .unicodeNumber:
            let n = value.unicodeScalars.first!.properties.numericValue.map { Int($0) } ?? 0
            return spokenNumber(String(n))
        case .longNumber:
            let label = #"(?i:phone(?: number)?|telephone(?: number)?|mobile(?: number)?|ZIP(?: code)?|postal(?: code)?|postcode)\s*(?:is)?\s*[:#]?\s*$"#
            if before(50).range(of: label, options: .regularExpression) != nil
                || value.range(of: #"^\+\d{1,3}[ -]+\d{5,}$"#, options: .regularExpression) != nil {
                return value.compactMap { c in c == "+" ? "plus" : c.wholeNumberValue.map { NumberWords.digits(String($0)) } }
                    .joined(separator: " ")
            }
            return spokenNumber(value)
        case .url, .technical, .identifier:
            return value
        }
    }
}

/// `EntityNormalizer.number`: inflect words, fractions as the grammar says them, e-notation, and
/// digit by digit past inflect's range.
func spokenNumber(_ text: String) -> String {
    var text = pythonStrip(text.replacingOccurrences(of: ",", with: ""))
    var sign = ""
    if let first = text.first, first == "-" || first == "+" {
        if first == "-" { sign = "negative " }
        text.removeFirst()
    }
    if let fraction = mixedFraction.fullMatch(text) {
        return sign + NumberWords.fraction(whole: fraction[1], numerator: fraction[2]!, denominator: fraction[3]!)
    }
    if let e = text.firstIndex(where: { $0 == "e" || $0 == "E" }) {
        return sign + spokenNumber(String(text[..<e])) + " times ten to the power of " + spokenNumber(String(text[text.index(after: e)...]))
    }
    if let words = NumberWords.inflect(text) { return sign + words }
    return sign + text.map { $0 == "." ? "point" : NumberWords.digits(String($0)) }.joined(separator: " ")
}

private let mixedFraction = Pattern(#"(?:(\d+)\s+)?(\d+)/(\d+)"#)

func spokenQuantity(_ text: String) -> String {
    if let pair = quantityPair.fullMatch(text) { return spokenNumber(pair[1]!) + " to " + spokenNumber(pair[2]!) }
    return spokenNumber(text)
}

private func plural(_ word: String, _ quantity: String) -> String {
    if word.hasSuffix("hertz") || ["yen", "won", "baht", "yuan"].contains(word) { return word }
    if let value = Decimal(string: quantity.replacingOccurrences(of: ",", with: ""), locale: Locale(identifier: "en_US_POSIX")),
       quantity.allSatisfy({ $0.isNumber || $0 == "." || $0 == "," || $0 == "-" || $0 == "+" }), abs(value) == 1 {
        return word
    }
    if word.hasPrefix("degree ") { return "degrees " + word.dropFirst("degree ".count) }
    if word.hasSuffix(" per minute") { return plural(String(word.dropLast(" per minute".count)), quantity) + " per minute" }
    if word == "millimeter of mercury" { return "millimeters of mercury" }
    if word.hasSuffix("foot") { return word.dropLast(4) + "feet" }
    if ["s", "x", "ch", "sh"].contains(where: word.hasSuffix) { return word + "es" }
    return word + "s"
}

private func moneyValue(_ symbol: String, _ value: String, _ scale: String?, original: String) -> String {
    var symbol = symbol, value = value
    if let sign = symbol.first, sign == "-" || sign == "+" {
        if value.first == "-" || value.first == "+" { return original }
        value = String(sign) + value
        symbol.removeFirst()
    }
    let name = plural(currencies[symbol] ?? symbol, scale == nil ? value : "2")
    return spokenQuantity(value) + (scale.flatMap { scaleWords[$0] }.map { " " + $0 } ?? "") + " " + name
}

private let monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                          "October", "November", "December"]

/// ISO and m/d/y dates. The year is read as the grammar reads years ("nineteen ninety"), where the
/// reference says "one thousand, nine hundred ninety".
private func spokenDate(_ text: String) -> String {
    let fields = text.split(whereSeparator: { "-/.".contains($0) })
    let parts = fields.map { NumberWords.value($0) }, yearFirst = fields[0].count == 4
    var (year, month, day) = yearFirst ? (parts[0], parts[1], parts[2]) : (parts[2], parts[0], parts[1])
    if !yearFirst && month > 12 { swap(&month, &day) }
    var calendar = Calendar(identifier: .gregorian)
    calendar.timeZone = TimeZone(identifier: "UTC")!
    let components = DateComponents(year: year, month: month, day: day)
    guard (1...9999).contains(year), calendar.date(from: components).map({ calendar.dateComponents([.year, .month, .day], from: $0) == components }) == true
    else { return text }
    return monthNames[month - 1] + " " + NumberWords.ordinal(String(day)) + ", " + spokenYear(year)
}

func spokenYear(_ year: Int) -> String {
    (1000...2999).contains(year) ? NumberWords.year(year) : NumberWords.cardinal(String(year))
}

private func spokenDateTime(_ text: String) -> String {
    let g = dateTime.fullMatch(text)!
    var value = spokenDate(g[1]!) + " at " + spokenClock(g[2]!, meridiem: nil)
    if let zone = g[3] {
        value += " UTC"
        if zone != "Z" {
            let hour = String(zone.dropFirst().prefix(2)), minute = String(zone.suffix(2))
            value += (zone.first == "+" ? " plus " : " minus ") + spokenNumber(hour) + " " + plural("hour", hour)
            if NumberWords.value(minute) != 0 { value += " " + spokenNumber(minute) + " " + plural("minute", minute) }
        }
    }
    return value
}

private let trailingMeridiem = try! NSRegularExpression(pattern: #"\s?([AaPp])\.?[Mm]\.?$"#)

/// One clock time or a range, with an optional am/pm after the last.
private func spokenTimes(_ text: String) -> String {
    let ns = text as NSString
    var clock = text, letter: String?
    if let m = trailingMeridiem.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) {
        clock = ns.substring(to: m.range.location)
        letter = ns.substring(with: m.range(at: 1)).uppercased()
    }
    let times = clock.components(separatedBy: CharacterSet(charactersIn: "-–—~～至到"))
        .map { pythonStrip($0) }.filter { !$0.isEmpty }
    return times.enumerated().map { i, t in spokenClock(t, meridiem: i == times.count - 1 ? letter : nil) }
        .joined(separator: " to ")
}

/// The grammar's clock: "three thirty", "eleven o five", "twelve o'clock", "five PM".
func spokenClock(_ text: String, meridiem: String?) -> String {
    let parts = text.split(separator: ":").map { NumberWords.value($0) }
    let hours = NumberWords.cardinal(String(parts[0]))
    let suffix = meridiem.map { " " + $0 + "M" } ?? ""
    if parts.count == 3 {
        let (m, s) = (parts[1], parts[2])
        return hours + " " + (parts[0] == 1 ? "hour" : "hours") + " " + NumberWords.belowHundred(m) + (m == 1 ? " minute" : " minutes")
            + " and " + NumberWords.belowHundred(s) + (s == 1 ? " second" : " seconds") + suffix
    }
    let minutes = parts.count > 1 ? parts[1] : 0
    if minutes == 0 { return hours + (meridiem == nil ? " o'clock" : suffix) }
    return hours + (minutes < 10 ? " o " : " ") + NumberWords.belowHundred(minutes) + suffix
}

/// "1990s" as "nineteen nineties", "'90s" as "nineties".
private func spokenDecade(_ text: String) -> String {
    let digits = text.filter(\.isNumber)
    let words = digits.count == 4 ? spokenYear(NumberWords.value(digits)) : NumberWords.belowHundred(NumberWords.value(digits))
    if words.hasSuffix("y") { return words.dropLast() + "ies" }
    return words + "s"
}

private func spokenVersion(_ text: String, context: String) -> String {
    let prefix = text.first == "v" || text.first == "V" ? String(text.first!) : ""
    let parts = text.dropFirst(prefix.count).split(separator: ".")
    if prefix.isEmpty && parts.count == 4 && parts.allSatisfy({ NumberWords.value($0) <= 255 })
        && context.range(of: #"(?i)version\s*(?:is)?\s*$"#, options: .regularExpression) == nil {
        return parts.map { spokenNumber(String($0)) }.joined(separator: " dot ")
    }
    return prefix + (prefix.isEmpty ? "" : " ") + parts.map { spokenNumber(String($0)) }.joined(separator: " point ")
}

private func spokenOperand(_ text: String) -> String {
    if text.hasPrefix("√") { return "square root of " + spokenOperand(pythonStrip(String(text.dropFirst()))) }
    if let power = powerPattern.fullMatch(text) {
        let exponent = power[2] ?? (power[3] == "²" ? "2" : "3")
        return spokenOperand(power[1]!) + " to the power of " + spokenNumber(exponent)
    }
    return numberPattern.fullMatch(text) != nil ? spokenNumber(text) : text
}

private func math(_ text: String, formula: Bool) -> String {
    let operators = ["+": "plus", "-": "minus", "*": "times", "×": "times", "÷": "divided by", "=": "equals",
                     ">": "greater than", "<": "less than", ">=": "greater than or equal to",
                     "<=": "less than or equal to", "≠": "not equal to", "≤": "less than or equal to",
                     "≥": "greater than or equal to", "≈": "approximately equals"]
    let ns = text as NSString
    guard let first = operandPattern.search.firstMatch(in: text, options: .anchored, range: NSRange(location: 0, length: ns.length))
    else { return text }
    var tokens = [ns.substring(with: first.range)], offset = first.range.length
    while offset < ns.length {
        guard let step = operatorStep.firstMatch(in: text, range: NSRange(location: offset, length: ns.length - offset))
        else { break }
        tokens += [ns.substring(with: step.range(at: 1)), ns.substring(with: step.range(at: 2))]
        offset += step.range.length
    }
    if !formula && tokens.count == 3 && tokens[1] == "-" {
        // A bare numeric hyphen is a range; F-5 or T-800 stays an identifier.
        return tokens.enumerated().allSatisfy({ $0.offset == 1 || numberPattern.fullMatch($0.element) != nil })
            ? spokenQuantity(text) : text
    }
    return tokens.enumerated().map { i, t in i % 2 == 0 ? spokenOperand(t) : operators[t]! }.joined(separator: " ")
}
