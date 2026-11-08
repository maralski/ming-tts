// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A memory-mapped safetensors file. Tensors are read by name as raw bytes; bf16 tensors are
/// copied into GPU buffers by `WeightArena`, fp32 ones are decoded on the host.
final class SafeTensors {
    struct Entry {
        let dtype: String
        let shape: [Int]
        let byteRange: Range<Int>
        var count: Int { shape.reduce(1, *) }
    }

    let entries: [String: Entry]
    private let data: Data
    private let dataStart: Int

    init(url: URL) throws {
        data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count >= 8 else { throw MingTTSError.badCheckpoint("file too small") }
        let headerLength = Int(data.withUnsafeBytes { $0.load(as: UInt64.self) })
        guard headerLength > 0, 8 + headerLength <= data.count else {
            throw MingTTSError.badCheckpoint("header length \(headerLength) exceeds file")
        }
        let header = data.subdata(in: 8..<(8 + headerLength))
        guard let json = try JSONSerialization.jsonObject(with: header) as? [String: Any] else {
            throw MingTTSError.badCheckpoint("header is not a JSON object")
        }
        dataStart = 8 + headerLength
        var entries: [String: Entry] = [:]
        for (name, value) in json where name != "__metadata__" {
            guard let dict = value as? [String: Any],
                  let dtype = dict["dtype"] as? String,
                  let shape = dict["shape"] as? [Int],
                  let offsets = dict["data_offsets"] as? [Int], offsets.count == 2 else {
                throw MingTTSError.badCheckpoint("malformed entry \(name)")
            }
            entries[name] = Entry(dtype: dtype, shape: shape, byteRange: offsets[0]..<offsets[1])
        }
        self.entries = entries
        if let last = entries.values.map(\.byteRange.upperBound).max(), dataStart + last > data.count {
            throw MingTTSError.badCheckpoint("file holds \(data.count) bytes, header needs \(dataStart + last); download incomplete?")
        }
    }

    func entry(_ name: String) throws -> Entry {
        guard let e = entries[name] else { throw MingTTSError.missingTensor(name) }
        return e
    }

    /// Calls `body` with the tensor's raw bytes.
    func withBytes<R>(_ name: String, _ body: (UnsafeRawBufferPointer, Entry) throws -> R) throws -> R {
        let e = try entry(name)
        return try data.withUnsafeBytes { raw in
            let slice = UnsafeRawBufferPointer(rebasing: raw[(dataStart + e.byteRange.lowerBound)..<(dataStart + e.byteRange.upperBound)])
            return try body(slice, e)
        }
    }

    /// Decodes a bf16 or f32 tensor to host floats.
    func floats(_ name: String) throws -> [Float] {
        try withBytes(name) { raw, e in
            switch e.dtype {
            case "BF16":
                let u = raw.bindMemory(to: UInt16.self)
                return u.map { Float(bitPattern: UInt32($0) << 16) }
            case "F32":
                return Array(raw.bindMemory(to: Float.self))
            default:
                throw MingTTSError.badCheckpoint("\(name) has dtype \(e.dtype)")
            }
        }
    }

    /// Row `index` of a bf16 [rows, width] table, decoded to floats.
    func bf16Row(_ name: String, index: Int) throws -> [Float] {
        try withBytes(name) { raw, e in
            guard e.dtype == "BF16", e.shape.count == 2, index < e.shape[0] else {
                throw MingTTSError.badCheckpoint("\(name) is not a bf16 table with row \(index)")
            }
            let width = e.shape[1]
            let u = raw.bindMemory(to: UInt16.self)
            return (0..<width).map { Float(bitPattern: UInt32(u[index * width + $0]) << 16) }
        }
    }
}
