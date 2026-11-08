// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The float32 initializers and node table of an ONNX file, read with a minimal protobuf decoder.
/// Only the fields CAM++ needs are decoded; everything else is skipped by wire type.
final class ONNXWeights {
    struct Tensor {
        let dims: [Int]
        let values: [Float]
    }

    private struct Node {
        let opType: String
        let inputs: [String]
        let floatAttributes: [String: Float]
    }

    private struct Initializer {
        let dims: [Int]
        let dataType: UInt64
        let rawData: Range<Int>?
    }

    private let data: Data
    private let nodes: [String: Node]
    private let initializers: [String: Initializer]

    init(url: URL) throws {
        data = try Data(contentsOf: url, options: .alwaysMapped)
        var nodes: [String: Node] = [:]
        var initializers: [String: Initializer] = [:]
        try data.withUnsafeBytes { raw in
            var model = ProtoReader(raw, 0..<raw.count)
            var graph: Range<Int>?
            while let (field, wire) = try model.next() {
                if field == 7, wire == 2 { graph = try model.bytes() } else { try model.skip(wire) }
            }
            guard let graph else { throw MingTTSError.badCheckpoint("ONNX model has no graph") }
            var g = ProtoReader(raw, graph)
            while let (field, wire) = try g.next() {
                switch (field, wire) {
                case (1, 2):
                    let (name, node) = try Self.parseNode(raw, try g.bytes())
                    nodes[name] = node
                case (5, 2):
                    let (name, tensor) = try Self.parseTensor(raw, try g.bytes())
                    initializers[name] = tensor
                default: try g.skip(wire)
                }
            }
        }
        self.nodes = nodes
        self.initializers = initializers
    }

    /// Weight and optional bias of the Conv node `name`.
    func conv(_ name: String) throws -> (weight: Tensor, bias: [Float]?) {
        let node = try node(name, op: "Conv")
        guard node.inputs.count >= 2 else { throw MingTTSError.badCheckpoint("Conv \(name) has no weight input") }
        let bias = node.inputs.count > 2 ? try tensor(node.inputs[2]).values : nil
        return (try tensor(node.inputs[1]), bias)
    }

    /// Per-channel eval-mode affine of the BatchNormalization node `name`. A node exported with
    /// affine=False has no scale or bias initializer, so they default to 1 and 0.
    func batchNorm(_ name: String) throws -> (scale: [Float], shift: [Float]) {
        let node = try node(name, op: "BatchNormalization")
        guard node.inputs.count == 5 else { throw MingTTSError.badCheckpoint("BatchNormalization \(name) has \(node.inputs.count) inputs") }
        let mean = try tensor(node.inputs[3]).values, variance = try tensor(node.inputs[4]).values
        let epsilon = node.floatAttributes["epsilon"] ?? 1e-5
        let affine = initializers[node.inputs[1]] != nil
        let weight = affine ? try tensor(node.inputs[1]).values : [Float](repeating: 1, count: mean.count)
        let bias = affine ? try tensor(node.inputs[2]).values : [Float](repeating: 0, count: mean.count)
        let scale = (0..<mean.count).map { weight[$0] / (variance[$0] + epsilon).squareRoot() }
        return (scale, (0..<mean.count).map { bias[$0] - mean[$0] * scale[$0] })
    }

    private func node(_ name: String, op: String) throws -> Node {
        guard let node = nodes[name] else { throw MingTTSError.badCheckpoint("campplus.onnx has no node \(name)") }
        guard node.opType == op else { throw MingTTSError.badCheckpoint("node \(name) is \(node.opType), expected \(op)") }
        return node
    }

    private func tensor(_ name: String) throws -> Tensor {
        guard let t = initializers[name] else { throw MingTTSError.badCheckpoint("campplus.onnx has no initializer \(name)") }
        let count = t.dims.reduce(1, *)
        guard t.dataType == 1, let range = t.rawData, range.count == count * 4 else {
            throw MingTTSError.badCheckpoint("initializer \(name) is not float32 raw data of \(t.dims)")
        }
        let values = data.withUnsafeBytes { raw in
            (0..<count).map { Float(bitPattern: raw.loadUnaligned(fromByteOffset: range.lowerBound + 4 * $0, as: UInt32.self)) }
        }
        return Tensor(dims: t.dims, values: values)
    }

    private static func parseNode(_ raw: UnsafeRawBufferPointer, _ range: Range<Int>) throws -> (String, Node) {
        var r = ProtoReader(raw, range)
        var name = "", opType = "", inputs: [String] = [], attributes: [String: Float] = [:]
        while let (field, wire) = try r.next() {
            switch (field, wire) {
            case (1, 2): inputs.append(r.string(try r.bytes()))
            case (3, 2): name = r.string(try r.bytes())
            case (4, 2): opType = r.string(try r.bytes())
            case (5, 2):
                var a = ProtoReader(raw, try r.bytes())
                var attrName = "", value: Float?
                while let (af, aw) = try a.next() {
                    switch (af, aw) {
                    case (1, 2): attrName = a.string(try a.bytes())
                    case (2, 5): value = Float(bitPattern: try a.fixed32())
                    default: try a.skip(aw)
                    }
                }
                if let value { attributes[attrName] = value }
            default: try r.skip(wire)
            }
        }
        return (name, Node(opType: opType, inputs: inputs, floatAttributes: attributes))
    }

    private static func parseTensor(_ raw: UnsafeRawBufferPointer, _ range: Range<Int>) throws -> (String, Initializer) {
        var r = ProtoReader(raw, range)
        var name = "", dims: [Int] = [], dataType: UInt64 = 0, rawData: Range<Int>?
        while let (field, wire) = try r.next() {
            switch (field, wire) {
            case (1, 0): dims.append(Int(try r.varint()))
            case (1, 2):
                var packed = ProtoReader(raw, try r.bytes())
                while !packed.atEnd { dims.append(Int(try packed.varint())) }
            case (2, 0): dataType = try r.varint()
            case (8, 2): name = r.string(try r.bytes())
            case (9, 2): rawData = try r.bytes()
            default: try r.skip(wire)
            }
        }
        return (name, Initializer(dims: dims, dataType: dataType, rawData: rawData))
    }
}

private struct ProtoReader {
    let raw: UnsafeRawBufferPointer
    var position: Int
    let end: Int

    init(_ raw: UnsafeRawBufferPointer, _ range: Range<Int>) {
        self.raw = raw
        position = range.lowerBound
        end = range.upperBound
    }

    var atEnd: Bool { position >= end }

    mutating func next() throws -> (field: Int, wire: Int)? {
        if atEnd { return nil }
        let key = try varint()
        return (Int(key >> 3), Int(key & 7))
    }

    mutating func varint() throws -> UInt64 {
        var value: UInt64 = 0, shift: UInt64 = 0
        while true {
            guard position < end, shift < 64 else { throw MingTTSError.badCheckpoint("truncated protobuf varint") }
            let byte = raw[position]
            position += 1
            value |= UInt64(byte & 0x7f) << shift
            if byte & 0x80 == 0 { return value }
            shift += 7
        }
    }

    mutating func fixed32() throws -> UInt32 {
        try advance(4)
        return raw.loadUnaligned(fromByteOffset: position - 4, as: UInt32.self)
    }

    mutating func bytes() throws -> Range<Int> {
        let length = Int(try varint())
        try advance(length)
        return (position - length)..<position
    }

    func string(_ range: Range<Int>) -> String {
        String(decoding: UnsafeRawBufferPointer(rebasing: raw[range]), as: UTF8.self)
    }

    mutating func skip(_ wire: Int) throws {
        switch wire {
        case 0: _ = try varint()
        case 1: try advance(8)
        case 2: _ = try bytes()
        case 5: try advance(4)
        default: throw MingTTSError.badCheckpoint("unsupported protobuf wire type \(wire)")
        }
    }

    private mutating func advance(_ count: Int) throws {
        guard count >= 0, position + count <= end else { throw MingTTSError.badCheckpoint("truncated protobuf field") }
        position += count
    }
}
