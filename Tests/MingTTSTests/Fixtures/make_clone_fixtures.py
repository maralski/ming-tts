# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Reference prompt ids for clone prompts across the task modes: writes clone_fixtures.json.

Each case is one build_input_ids call with a prompt_text and prompt_latent_steps, with and without
the speaker row (use_spk_emb), the fragment sent as given (raw text, no normalization).

    python make_clone_fixtures.py REF
"""
import json, os, sys

ref = os.path.expanduser(sys.argv[1])
sys.path.insert(0, ref + "/code")
from transformers import AutoTokenizer
import prompt_contract, tts_tasks

tok = AutoTokenizer.from_pretrained(ref + "/weights")
prompt_contract.validate_tokenizer(tok, "dense")

prompts = [
    ("Markets rallied this morning after the central bank held interest rates steady.", 62),
    ("诶，你今天有空吗？我们能不能聊聊这件事啊？", 42),
    ("A short one.", 1),
]
texts = [
    "The library extends its opening hours during the examination period.",
    "雨后的山谷很安静，远处亮起了一盏温暖的灯。",
    "[Laughter] That was funny, really. [Sigh] Honestly.",
    "Genre: rock Mood: calm Instrument: piano Theme: x Duration: 3",
]
tasks = [
    ("TTS", None, None, "en"), ("TTS", None, None, None), ("MULTILINGUAL", None, None, "ja"),
    ("IP", {"ip": "US_Emma"}, None, "en"), ("VOICE_DESIGN", {"style": "A gravelly old sea captain."}, None, "en"),
    ("FREEFORM", None, "Speak softly.", "en"), ("EMOTION", {"emotion": "高兴"}, None, "en"),
    ("BASIC", {"speed": "慢速", "volume": "高"}, None, "zh"), ("DIALECT", {"dialect": "cantonese"}, None, None),
]

cases = []
for task, controls, instruction, language in tasks:
    for text in texts:
        for prompt_text, steps in prompts:
            for use_spk in (True, False):
                req = tts_tasks.build_task_generation_kwargs(
                    task, text=text, language_id=language, controls=controls, instruction=instruction,
                    prompt_wav_path="ref.wav", prompt_text=prompt_text, use_spk_emb=use_spk)
                ids, spk, latent = prompt_contract.build_input_ids(
                    tok, "dense", req["prompt"], req["text"], num_speakers=1 if use_spk else 0,
                    instruction=req["instruction"], prompt_text=prompt_text, prompt_latent_steps=steps)
                cases.append({
                    "task": task, "controls": controls or {}, "instruction": instruction, "language": language,
                    "text": text, "prompt_text": prompt_text, "latent_steps": steps, "speaker_embedding": use_spk,
                    "ids": ids, "speaker_positions": spk, "latent_position": latent,
                })

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "clone_fixtures.json")
json.dump(cases, open(out, "w"), ensure_ascii=False)
print(len(cases), "cases", os.path.getsize(out), "bytes")
