# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Reference prompts for every task mode, for TaskTests: writes task_fixtures.json.

Each case is one MingTTS.synthesize request through the reference: build_task_generation_kwargs,
prepare_text (TalkerTN normalization), split_text, then build_input_ids per fragment.

Two deliberate differences from the reference are encoded here, so the fixture states what the
Swift port must produce:
  - English text is normalized by the Swift port, which keeps the written space between
    punctuation and a paralinguistic tag; TaskTests lists those cases.
  - Chinese normalization is not ported. Where the reference would run Chinese TN (language zh,
    or no language and a Han character in the text) the fixture is the text_normalization=False
    path. "reference_normalizer" says which path a case took.

Needs pynini (see make_normalize_fixtures.py) and transformers in the same environment:
    ./tnenv/bin/pip install inflect==7.6.0 transformers==4.46.3
    ./tnenv/bin/python make_task_fixtures.py REF
"""
import json, os, re, sys

ref = os.path.expanduser(sys.argv[1])
sys.path.insert(0, ref + "/code")
from transformers import AutoTokenizer
import prompt_contract, tts_tasks
from frontend.talker_tn.talker_tn import TalkerTN
from frontend.toolkit import normalize_text_kit
from frontend.text_processing import prepare_text, split_text

tok = AutoTokenizer.from_pretrained(ref + "/weights")
prompt_contract.validate_tokenizer(tok, "dense")
tn = TalkerTN()
normalizer = lambda text, language=None: normalize_text_kit(tn.normalize(text, language=language))
han = re.compile(r"[\u3400-\u9fff\U00020000-\U0003134f]")
kana = re.compile(r"[\u3040-\u30ff]")

texts = [
    "The garden will remain open until sunset, provided the weather stays clear.",
    "Officials in the coastal city said on Tuesday that the new flood barrier had held through the overnight storm. Engineers will inspect the structure this week.",
    "Dr. Smith paid $3.5bn on Nov 6, up 25% from 2026.",
    "[Laughter] That was funny, really[Sigh]honestly. Well [Uhm] I think [Breathing] so? [Question-huh]",
    "# Notice\n**Bold** news: see [the site](https://x.com/a_(b)) and `code` [Cough](aside) now",
    "週末、駅前の小さな喫茶店で、季節限定の抹茶パフェを食べました",
    "Am Wochenende gehen wir mit 3 Freunden ins Kino, oder?",
    "今日天气几好，我哋食完饭一齐去海边行下啦。",
    "[Sigh]雨还是没有停，[Cough]我们恐怕只能把活动改到室内了。",
]

# (reference task, controls, instruction, language_id)
tasks = [
    ("TTS", None, None, None),
    ("TTS", None, None, "en"),
    ("TTS", None, None, "zh"),
    ("MULTILINGUAL", None, None, "ja"),
    ("MULTILINGUAL", None, None, "de"),
    ("IP", {"ip": "UK_Charlotte"}, None, "en"),
    ("IP", {"ip": "灵小柔"}, None, "zh"),
    ("IP", {"ip": "US_Emma"}, None, None),
    ("VOICE_DESIGN", {"style": "A gravelly old sea captain, slow and amused."}, None, "en"),
    ("VOICE_DESIGN", {"style": "成熟沉稳的男性纪录片旁白，吐字清晰。"}, None, None),
    ("FREEFORM", None, "  Speak in a warm, intimate storytelling voice, unhurried.  ", "en"),
    ("EMOTION", {"emotion": "高兴"}, None, "en"),
    ("EMOTION", {"emotion": "悲伤"}, None, None),
    ("BASIC", {"speed": "慢速"}, None, "en"),
    ("BASIC", {"speed": "快速", "pitch": "低", "volume": "高"}, None, "zh"),
    ("DIALECT", {"dialect": "cantonese"}, None, None),
    ("DIALECT", {"dialect": "chuanyu dialect"}, None, None),
]

cases = []
for task, controls, instruction, language in tasks:
    for text in texts:
        if language == "en" and kana.search(text):
            continue  # English TN over Japanese also folds kanji to simplified Chinese; the port does not.
        for zero in (True, False):
            req = tts_tasks.build_task_generation_kwargs(
                task, text=text, language_id=language, controls=controls, instruction=instruction,
                use_zero_spk_emb=zero)
            chinese_tn = language in ("zh",) or (language is None and han.search(text))
            prepared = prepare_text(req["text"], task=req["taskname"], language=req["text_language"],
                                    normalizer=None if chinese_tn else normalizer)
            fragments = split_text(prepared, 24)
            prompts = []
            for fragment in fragments:
                ids, spk, latent = prompt_contract.build_input_ids(
                    tok, "dense", req["prompt"], fragment,
                    num_speakers=1 if req["use_zero_spk_emb"] else 0, instruction=req["instruction"])
                assert latent == -1
                prompts.append({"ids": ids, "speaker_positions": spk})
            cases.append({
                "task": task, "controls": controls or {}, "instruction": instruction, "language": language,
                "speaker": "zero" if zero else "none", "text": text, "reference_normalizer": not chinese_tn,
                "fragments": fragments, "prompts": prompts,
            })

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "task_fixtures.json")
json.dump(cases, open(out, "w"), ensure_ascii=False)
print(len(cases), "cases", sum(len(c["prompts"]) for c in cases), "prompts", os.path.getsize(out), "bytes")
