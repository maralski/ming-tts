// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
@testable import MingTTS

/// The checkpoint directory (MING_WEIGHTS) and the reference goldens directory (MING_GOLDENS).
/// Tests that need either are disabled when it is unset.
let weightsDirectory = ProcessInfo.processInfo.environment["MING_WEIGHTS"].map { URL(fileURLWithPath: $0) }
let goldensDirectory = ProcessInfo.processInfo.environment["MING_GOLDENS"].map { URL(fileURLWithPath: $0) }

let testTokenizer: MingTokenizer = try! MingTokenizer(tokenizerJSON: weightsDirectory!.appending(path: "tokenizer.json"))
