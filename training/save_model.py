import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from read_book import read_books, build_vocab_and_bigram, folder_path, files

all_readed_tokens = read_books(folder_path, files)
clean_count, bigram = build_vocab_and_bigram(all_readed_tokens)

script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "..", "models")
os.makedirs(models_dir, exist_ok=True)

with open(os.path.join(models_dir, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(clean_count, f, ensure_ascii=False, indent=2)

with open(os.path.join(models_dir, "bigram.json"), "w", encoding="utf-8") as f:
    json.dump(bigram, f, ensure_ascii=False, indent=2)

print("模型已保存到 models/")
