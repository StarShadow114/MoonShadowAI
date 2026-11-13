import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from read_book import build_vocab_and_bigram

script_dir = os.path.dirname(os.path.abspath(__file__))
materials_dir = os.path.join(script_dir, "materials")

vocab, bigram = build_vocab_and_bigram(materials_dir)

models_dir = os.path.join(script_dir, "..", "models")
os.makedirs(models_dir, exist_ok=True)

with open(os.path.join(models_dir, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

with open(os.path.join(models_dir, "bigram.json"), "w", encoding="utf-8") as f:
    json.dump(bigram, f, ensure_ascii=False, indent=2)

print("模型已保存到 models/")
