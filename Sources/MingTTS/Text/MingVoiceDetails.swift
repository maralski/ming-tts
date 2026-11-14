// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// What a built-in voice's name says about it. The cookbook writes an English voice as
/// `US_Emma`, a Chinese voice recorded in one emotion as `<voice>_<emotion>`, a character from a
/// show as `<show>_<character>`, and any other Chinese voice as a bare name.
extension MingVoice {
    public enum Kind: Sendable, Hashable, CaseIterable {
        /// A Chinese voice with a bare name, such as `灵小安`.
        case chinese
        /// A Chinese voice recorded in one emotion, such as `灵妩月_高兴`.
        case chineseEmotion
        /// A character from a show, such as `西游记_唐僧`.
        case chineseCharacter
        /// An English voice, such as `UK_George`.
        case english
    }

    public var kind: Kind {
        guard let (prefix, _) = parts else { return .chinese }
        if prefix.allSatisfy(\.isASCII) { return .english }
        return emotion == nil ? .chineseCharacter : .chineseEmotion
    }

    /// The language the voice was recorded in.
    public var language: MingLanguage { kind == .english ? .english : .chinese }

    /// The emotion a `<voice>_<emotion>` voice was recorded in. The cookbook names angry voices
    /// `生气`, which is not `MingEmotion.angry`'s raw value.
    public var emotion: MingEmotion? {
        guard let suffix = parts?.suffix else { return nil }
        return suffix == "生气" ? .angry : MingEmotion(rawValue: suffix)
    }

    private var parts: (prefix: Substring, suffix: String)? {
        let split = ip.split(separator: "_", maxSplits: 1)
        return split.count == 2 ? (split[0], String(split[1])) : nil
    }
}
