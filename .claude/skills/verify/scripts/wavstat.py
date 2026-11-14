#!/usr/bin/env python3
"""Prints facts about a WAV file as JSON: format, length, peak, RMS, silence at each end.

    .claude/skills/verify/scripts/wavstat.py out.wav

ming-tts writes 32-bit float mono at 44,100 Hz, which Python's wave module cannot read,
so this parses the RIFF chunks itself. It also reads 16-bit PCM, for reference clips.
"""
import array
import json
import math
import struct
import sys

SILENCE = 1e-3  # |sample| below this counts as silence


def read(path):
    data = open(path, "rb").read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        sys.exit(f"{path} is not a RIFF/WAVE file")
    fmt = None
    samples = None
    at = 12
    while at + 8 <= len(data):
        tag, size = data[at:at + 4], struct.unpack("<I", data[at + 4:at + 8])[0]
        body = data[at + 8:at + 8 + size]
        if tag == b"fmt ":
            code, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            fmt = {"format": {1: "pcm", 3: "float"}.get(code, str(code)), "channels": channels, "sample_rate": rate, "bits": bits}
        elif tag == b"data":
            if fmt is None:
                sys.exit("data chunk before fmt chunk")
            if fmt["format"] == "float" and fmt["bits"] == 32:
                samples = array.array("f")
                samples.frombytes(body[:len(body) // 4 * 4])
            elif fmt["format"] == "pcm" and fmt["bits"] == 16:
                ints = array.array("h")
                ints.frombytes(body[:len(body) // 2 * 2])
                samples = array.array("f", (x / 32768 for x in ints))
            else:
                sys.exit(f"unsupported sample format {fmt}")
        at += 8 + size + (size & 1)
    if fmt is None or samples is None:
        sys.exit("no fmt or data chunk")
    return fmt, samples


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    fmt, samples = read(sys.argv[1])
    frames = len(samples) // fmt["channels"]
    rate = fmt["sample_rate"]
    nonfinite = sum(1 for x in samples if not math.isfinite(x))
    finite = [x for x in samples if math.isfinite(x)]
    peak = max((abs(x) for x in finite), default=0.0)
    rms = math.sqrt(sum(x * x for x in finite) / len(finite)) if finite else 0.0
    lead = next((i for i, x in enumerate(samples) if abs(x) >= SILENCE), len(samples))
    trail = next((i for i, x in enumerate(reversed(samples)) if abs(x) >= SILENCE), len(samples))
    print(json.dumps({
        **fmt,
        "frames": frames,
        "seconds": round(frames / rate, 3),
        "peak": round(peak, 4),
        "rms": round(rms, 4),
        "nonfinite_samples": nonfinite,
        "leading_silence_seconds": round(lead / fmt["channels"] / rate, 3),
        "trailing_silence_seconds": round(trail / fmt["channels"] / rate, 3),
    }, indent=2))


main()
