# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Reference English normalization for NormalizeTests.

Runs MingTTS.prepare_text's text path for English (TalkerTN, the en_tn FSTs and the entity rules,
then normalize_text_kit and add_period) over normalize_cases.txt (hand-written),
normalize_synthetic.txt (synthetic news-style sentences) and synthetic_bulletin.txt (a synthetic
spoken bulletin, one paragraph per line), and writes normalize_fixtures.json.

TalkerTN needs pynini, which has no macOS wheel on PyPI; conda-forge has one:
    mamba create -p ./tnenv -c conda-forge python=3.11 pynini=2.1.6
    ./tnenv/bin/pip install inflect==7.6.0
    ./tnenv/bin/python make_normalize_fixtures.py REF/code
"""
import json, os, sys

code = os.path.expanduser(sys.argv[1])
sys.path.insert(0, code)
from frontend.talker_tn.talker_tn import TalkerTN
from frontend.toolkit import normalize_text_kit
from frontend.text_processing import prepare_text

here = os.path.dirname(os.path.abspath(__file__))
tn = TalkerTN()
normalizer = lambda text, language=None: normalize_text_kit(tn.normalize(text, language=language))
cases = []
for name in ("normalize_cases.txt", "normalize_synthetic.txt", "synthetic_bulletin.txt"):
    for line in open(os.path.join(here, name), encoding="utf-8").read().split("\n"):
        if line.strip():
            cases.append({"text": line, "reference": prepare_text(line, task="IP", language="en", normalizer=normalizer)})
json.dump(cases, open(os.path.join(here, "normalize_fixtures.json"), "w"), ensure_ascii=False, indent=0)
print(len(cases))
