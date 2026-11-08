// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

public enum MingTTSError: Error, CustomStringConvertible {
    case noMetalDevice
    case shaderCompilation(String)
    case missingKernel(String)
    case gpuFailure(String)
    case badCheckpoint(String)
    case missingTensor(String)
    case badInput(String)

    public var description: String {
        switch self {
        case .noMetalDevice: return "no Metal device"
        case .shaderCompilation(let s): return "shader compilation failed: \(s)"
        case .missingKernel(let s): return "missing kernel \(s)"
        case .gpuFailure(let s): return "GPU failure: \(s)"
        case .badCheckpoint(let s): return "bad checkpoint: \(s)"
        case .missingTensor(let s): return "missing tensor \(s)"
        case .badInput(let s): return "bad input: \(s)"
        }
    }
}

/// Owns the device, the queue, the shader library compiled from `ShaderSource` and the pipelines.
final class MetalContext: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    init(device: MTLDevice? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else { throw MingTTSError.noMetalDevice }
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw MingTTSError.noMetalDevice }
        self.queue = queue
        let options = MTLCompileOptions()
        // Safe math keeps exp/-inf semantics the softmax relies on and matches the fp32 reference closely.
        if #available(macOS 15.0, *) {
            options.mathMode = .safe
        } else {
            options.fastMathEnabled = false
        }
        options.languageVersion = .version3_1
        do {
            self.library = try device.makeLibrary(source: ShaderSource.all, options: options)
        } catch {
            throw MingTTSError.shaderCompilation(String(describing: error))
        }
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock()
        if let p = pipelines[name] { lock.unlock(); return p }
        lock.unlock()
        guard let function = library.makeFunction(name: name) else { throw MingTTSError.missingKernel(name) }
        let p: MTLComputePipelineState
        do { p = try device.makeComputePipelineState(function: function) } catch {
            throw MingTTSError.shaderCompilation("pipeline \(name): \(error)")
        }
        lock.lock()
        pipelines[name] = p
        lock.unlock()
        return p
    }

    func warmUpPipelines() throws {
        for name in library.functionNames { _ = try pipeline(name) }
    }

    func makeBuffer(bytes: Int, label: String) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
            throw MingTTSError.gpuFailure("cannot allocate \(bytes) bytes for \(label)")
        }
        b.label = label
        memset(b.contents(), 0, b.length)
        return b
    }

    // Concrete element types on purpose: a generic throwing variant called through @testable
    // import left the Swift error register dirty in release builds (Swift 6.2.4) and crashed XCTest.
    func makeBuffer(floats: [Float], label: String) throws -> MTLBuffer {
        let b = try makeBuffer(bytes: floats.count * 4, label: label)
        b.write(floats)
        return b
    }

    func makeBuffer(bf16: [UInt16], label: String) throws -> MTLBuffer {
        let b = try makeBuffer(bytes: bf16.count * 2, label: label)
        _ = bf16.withUnsafeBytes { memcpy(b.contents(), $0.baseAddress!, $0.count) }
        return b
    }
}

extension MTLBuffer {
    func floats(count: Int, offset: Int = 0) -> [Float] {
        let p = (contents() + offset).bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }

    func write(_ values: [Float], offset: Int = 0) {
        precondition(offset + values.count * 4 <= length)
        values.withUnsafeBytes { raw in
            if let base = raw.baseAddress { memcpy(contents() + offset, base, raw.count) }
        }
    }
}
