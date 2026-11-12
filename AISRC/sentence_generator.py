import json
import random
import os

# 1. 定位模型路径（AISRC 和 models 同级，往上跳一级）
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "..", "models")


def load_models():
    """加载两个 JSON，返回 (vocab, bigram)"""
    # TODO: 用 json.load 读两个文件
    with open(os.path.join(models_dir, "vocab.json"), "r", encoding="utf-8") as rvoc:
        vocab = json.load(rvoc)
    with open(os.path.join(models_dir, "bigram.json"), "r", encoding="utf-8") as rbr:
        bigram = json.load(rbr)
    return vocab, bigram


def pick_next(next_words):
    """给一个 {"喜欢": 3, "爱": 1}，按权重返回一个词"""
    # TODO: 方案A 手写累加 / 方案B random.choices
    words = list(next_words.keys())
    weights = list(next_words.values())
    return random.choices(words, weights=weights, k=1)[0]


def generate(start_word, max_length=20):
    """从 start_word 开始生成一句话"""
    vocab, bigram = load_models()

    # TODO: 校验 start_word 在不在 vocab 里、在不在 bigram 里
    # TODO: 上面那个循环
    # TODO: 用 "".join() 拼成字符串返回
    pass


if __name__ == "__main__":
    for word in ["学习", "数据", "项目"]:
        print(word, "→", generate(word))
