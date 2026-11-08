// swift-tools-version: 6.0
// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import PackageDescription

let package = Package(
    name: "MingTTS",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MingTTS", targets: ["MingTTS"]),
        .executable(name: "ming-tts", targets: ["ming-tts"]),
    ],
    targets: [
        .target(name: "MingTTS", path: "Sources/MingTTS"),
        .executableTarget(name: "ming-tts", dependencies: ["MingTTS"], path: "Sources/ming-tts"),
        .testTarget(name: "MingTTSTests", dependencies: ["MingTTS"], path: "Tests/MingTTSTests", resources: [.copy("Fixtures")]),
    ]
)
