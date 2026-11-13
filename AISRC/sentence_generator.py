import json
import random
import os

# 1. 定位模型路径（AISRC 和 models 同级，往上跳一级）
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "..", "models")


def load_models():
    """加载两个 JSON，返回 (vocab, bigram)"""
    with open(os.path.join(models_dir, "vocab.json"), "r", encoding="utf-8") as rvoc:
        vocab = json.load(rvoc)
    with open(os.path.join(models_dir, "bigram.json"), "r", encoding="utf-8") as rbr:
        bigram = json.load(rbr)
    return vocab, bigram


def pick_next(next_words):
    """给一个 {"喜欢": 3, "爱": 1}，按权重返回一个词"""
    words = list(next_words.keys())
    weights = list(next_words.values())
    return random.choices(words, weights=weights, k=1)[0]


def generate(start_word, max_length=20):
    """从 start_word 开始生成一句话"""
    vocab, bigram = load_models()
    if start_word not in vocab:
        return f"'{start_word}' 不在词表中"
    if start_word not in bigram:
        return f"'{start_word}' 后面没有接任何词"
    result = [start_word]
    current = start_word
    for _ in range(max_length - 1):
        if current not in bigram:
            break
        next_words = bigram[current]
        if not next_words:
            break
        next_word = pick_next(next_words)
        result.append(next_word)
        current = next_word
    return "".join(result) + "。"


if __name__ == "__main__":
    while True:
        word = input("MoonShadowAI. Please input a start word(pass 'exit' to exit.): ")
        if word == "exit":
            break
        else:
            print(word, "→", generate(word))
