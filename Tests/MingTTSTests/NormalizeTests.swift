// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import MingTTS

/// `reference` is the reference frontend's English text path (TalkerTN), from
/// Fixtures/make_normalize_fixtures.py. normalize_deviations.json holds the reviewed cases where
/// the Swift normalizer reads text differently on purpose, because the reference grammar garbles
/// it: "1990s" as seconds, "52-48 in" as inches, "3:30pm" as "thirtypm", "$950m" as meters,
/// "#LanternFest2026" as "numberLanternFest2026", "1,500" as "thousand five hundred", "per match" as "per - minute atch".
private struct NormalizeCase: Decodable { var text: String; var reference: String }

private let fixtures: [NormalizeCase] = {
    let url = Bundle.module.url(forResource: "normalize_fixtures", withExtension: "json", subdirectory: "Fixtures")!
    return try! JSONDecoder().decode([NormalizeCase].self, from: Data(contentsOf: url))
}()

private let deviations: [String: String] = {
    let url = Bundle.module.url(forResource: "normalize_deviations", withExtension: "json", subdirectory: "Fixtures")!
    return try! JSONDecoder().decode([String: String].self, from: Data(contentsOf: url))
}()

/// `prepare_text` for English up to `split_text`.
private func prepared(_ text: String) -> String {
    addPeriod(normalizeEnglish(pythonStrip(stripMarkdown(text))))
}

@Test func normalizeEnglishMatchesReferenceOrReviewedDeviation() {
    let failures = fixtures.filter { prepared($0.text) != (deviations[$0.text] ?? $0.reference) }
    for failure in failures.prefix(5) {
        Issue.record("\(failure.text)\n got  \(prepared(failure.text))\n want \(deviations[failure.text] ?? failure.reference)")
    }
    let parity = fixtures.filter { deviations[$0.text] == nil }.count
    print("normalizeEnglish: \(parity) of \(fixtures.count) cases read as the reference does")
    #expect(failures.isEmpty, "\(failures.count) of \(fixtures.count) cases differ")
}

@Test func normalizeEnglishDeviationsStayReviewed() {
    let references = Dictionary(fixtures.map { ($0.text, $0.reference) }, uniquingKeysWith: { a, _ in a })
    let stale = deviations.filter { references[$0.key] == nil || references[$0.key] == $0.value }
    #expect(stale.isEmpty, "deviations without a differing reference case: \(stale.keys.prefix(3))")
}

@Test func normalizeEnglishReadsNewsEntities() {
    #expect(prepared("The deal is worth $3.5bn, up 25% from 2026.")
            == "The deal is worth three point five billion dollars, up twenty five percent from twenty twenty six.")
    #expect(prepared("Meet at 3:30pm on Nov 6 in the U.S.") == "Meet at three thirty PM on the sixth of november in the US.")
    #expect(prepared("He came 1st at 120 km/h.") == "He came first at one hundred twenty kilometers per hour.")
    #expect(prepared("[Laughter] It cost £12.50 [Sigh]") == "[Laughter] It cost twelve point five zero pounds [Sigh]")
}

/// A spoken bulletin runs about 400 words; this normalizes it in about 30 ms on an M3 Max.
@Test func normalizeEnglishIsFastOnABulletin() {
    let bulletin = fixtures.prefix(50).map(\.text).joined(separator: " ")
    _ = normalizeEnglish(bulletin)
    let elapsed = ContinuousClock().measure { _ = normalizeEnglish(bulletin) }
    print("normalizeEnglish \(bulletin.split(separator: " ").count) words: \(elapsed)")
    #expect(elapsed < .milliseconds(100))
}

@Test func normalizeEnglishRunsBeforeSplitting() {
    let request = try! MingRequest(text: "Shares fell 2.5% at 3:30pm.", task: .builtInVoice(.emma), language: .english, conditioning: .zeroSpeaker)
    #expect(try! prepareFragments(request, maxLength: 24) == ["Shares fell two point five percent at three thirty PM."])
    #expect(prepareText(request.text, task: request.task, language: .english, normalizing: false) == "Shares fell 2.5% at 3:30pm.")
}
