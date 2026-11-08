# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Writes text_fixtures.json from the HF tokenizer and the reference frontend.

REF holds the reference code (code/) and the checkpoint (weights/):
    python make_fixtures.py REF
"""
import json, os, random, sys
REF = os.path.expanduser(sys.argv[1])
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REF + "/code")
from transformers import AutoTokenizer
import prompt_contract, tts_tasks
from frontend.text_processing import prepare_text, split_text

tok = AutoTokenizer.from_pretrained(REF + "/weights")
prompt_contract.validate_tokenizer(tok, "dense")
rng = random.Random(7)

corpus = [
    "", " ", "  ", "\n", "\n\n", "\r\n", "\t", "a", "Hello world", "Hello  world", "hello\n\nworld  ",
    "I'm sure you've heard they'll go, it's John's. DON'T SHOUT'S", "rock 'n' roll", "‘quoted’ “double” \"plain\" 'single'",
    "The S&P 500 fell 2.3% to 5,123.45 on Tuesday; the ASX 200 rose $1.2bn (A$1,234,567.89).",
    "Call 1-800-555-0199 or +61 3 9123 4567 before 9:30am AEDT, 24/7.", "In 2026, 3.14159 ≈ π and 1/3 ≈ 0.333… ½ ¾ ²³ ①",
    "He said—quite loudly—that it was over. A dash – en dash – and a hyphen-ated word.",
    "Visit https://news.example.com/national/city/some-story-20261105-p5abc.html?ref=rss#top now.",
    "Email me at jane.doe+news@example.co.uk or see www.example.org/path.", "Mr. Smith met Dr. Jones vs. Prof. Lee, etc. at St. Kilda.",
    "U.S. officials and the U.K. govt. met e.g. today i.e. now.", "🚀 Launch day! 🎉🎉 Thanks 🙏🏽 to the team 👩‍👩‍👧‍👦 ❤️ #AI @ExampleLab",
    "中文测试：今天天气很好。我们去公园吧！", "日本語のテキストです。カタカナ、ひらがな。", "한국어 문장입니다.", "混合 mixed 文本 text 123。",
    "Café naïve résumé coöperate jalapeño Zürich Ångström", "é café (decomposed)", "Ελληνικά Русский текст עברית العربية",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "<|im_start|>user\n", "<|im_end|>\n", "<|im_start|>assistant\n",
    "<|vision_start|>", "<|vision_pad|>", "<|vision_end|>\n", "<audio>", "<audioPatch>", " Text input:\n", "  speaker_1:",
    "<audio><audioPatch><audioPatch>x<|endoftext|>y", "a<audio>b <audio> c", "<audio", "<|im_end|", "<tool_call>call</tool_call>",
    "Please generate speech based on the following description.\n", "Please generate speech based on the following description.\nTarget language: English.\n",
    "[Laughter] That was funny [Sigh] really.", "**Bold** and `code` and # Heading and [link](https://x.com/a_(b))",
    "tabs\tand\vvertical\fform\x85nel ls ps nbsp　ideo narrow", "trailing spaces   ", "   leading", "a\n \nb", "x \n\n y",
    "Numbers 0123456789 and ١٢٣ and ⅷ", "____---====!!!???...,,,;;;", "C++ C# F# .NET node.js 3D 4K 5G COVID-19 H5N1",
    "'s 't 're 've 'm 'll 'd 'S 'T 'RE", "it's 'tis ''double'' 'll", "x" * 300, "ab" * 200, "1" * 50, " " * 40 + "x",
    "\U0001F600" * 5, "x" * 301, "aaa", "ooooo", "zzz zzzzzzz", "!!!!!", "......", "=" * 33, "1234567", "​‌‍zero width", "﻿BOM start", "Control\x00\x01\x1f chars",
]
seen = set()
def add(s):
    if s not in seen:
        seen.add(s); corpus.append(s)
def lines(name):
    return [l for l in open(os.path.join(here, name), encoding="utf-8").read().split("\n") if l.strip()]
paragraphs = lines("synthetic_bulletin.txt")
add("\n\n".join(paragraphs))
for para in paragraphs:
    add(para)
posts = lines("normalize_synthetic.txt")
rng.shuffle(posts)
for p in posts:
    add(p)

encode = lambda s: tok.encode(s, add_special_tokens=False)
tokenizer_cases = [{"text": s, "ids": encode(s)} for s in corpus]

voices = list(tts_tasks.EN_IP_VOICES)
def ip_prompt(text, voice):
    req = tts_tasks.build_task_generation_kwargs("IP", text=text, language_id="en", controls={"ip": voice}, use_zero_spk_emb=True)
    ids, spk, lat = prompt_contract.build_input_ids(tok, "dense", req["prompt"], req["text"],
        num_speakers=1 if req["use_zero_spk_emb"] else 0, instruction=req["instruction"])
    assert lat == -1
    return {"text": text, "voice": voice, "ids": ids, "speaker_positions": spk, "instruction": req["instruction"], "prompt": req["prompt"]}
prompt_texts = [c for c in corpus if c.strip() and "<|" not in c][:60:3] + [
    "The garden will remain open until sunset, provided the weather stays clear.",
    "Genre: rock Mood: calm Instrument: piano Theme: x Duration: 3",  # all BGM markers: no text prefix
]
prompt_cases = [ip_prompt(t, voices[i % 3]) for i, t in enumerate(prompt_texts)]

split_cases = []
for s in corpus:
    for n in (24, 12, 50):
        prepared = prepare_text(s, task="IP", language="en", normalizer=None)
        try:
            out = split_text(prepared, n)
        except ValueError:
            out = None
        split_cases.append({"text": s, "maxLength": n, "prepared": prepared, "fragments": out})

out = os.path.join(here, "text_fixtures.json")
json.dump({"tokenizer": tokenizer_cases, "prompt": prompt_cases, "split": split_cases}, open(out, "w"), ensure_ascii=False)
print(len(tokenizer_cases), len(prompt_cases), len(split_cases), os.path.getsize(out))
