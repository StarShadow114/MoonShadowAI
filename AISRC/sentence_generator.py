import os
import sys
import random
import struct

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
models_dir = os.path.join(root_dir, "models")


def load_vocab_binary(filepath):
    """从二进制词表文件加载：返回 {编号: 词} 和 {词: 编号}"""
    id_to_word = {}
    word_to_id = {}
    with open(filepath, "rb") as f:
        while True:
            header = f.read(6)
            if not header or len(header) < 6:
                break
            idx, length = struct.unpack("<IH", header)
            content = f.read(length)
            word = content.decode("utf-8")
            id_to_word[idx] = word
            word_to_id[word] = idx
    return id_to_word, word_to_id


def load_bigram_binary(filepath):
    """从二进制 bigram 文件加载：返回 {源词: {目标词: 权重}}"""
    bigram = {}
    with open(filepath, "rb") as f:
        while True:
            record = f.read(10)
            if not record or len(record) < 10:
                break
            first_id, second_id, weight, score = struct.unpack("<IIBB", record)
            first = id_to_word.get(first_id, "")
            second = id_to_word.get(second_id, "")
            if not first or not second:
                continue
            if first not in bigram:
                bigram[first] = {}
            bigram[first][second] = weight
    return bigram


def load_models():
    """加载词表和 bigram，返回 (vocab_set, bigram_dict)"""
    global id_to_word
    vocab_path = os.path.join(models_dir, "vocab.mobu")
    bigram_path = os.path.join(models_dir, "bigram.mobu")

    id_to_word, word_to_id = load_vocab_binary(vocab_path)
    vocab = set(id_to_word.values())
    bigram = load_bigram_binary(bigram_path)
    return vocab, bigram


id_to_word = {}


def pick_next(next_words):
    """根据权重随机选取下一个词"""
    words = list(next_words.keys())
    weights = list(next_words.values())
    return random.choices(words, weights=weights, k=1)[0]


def generate(start_word, max_length=20):
    """从起始词开始生成句子"""
    vocab, bigram = load_models()

    if start_word not in vocab:
        return f"'{start_word}' 不在词表里"
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
    print("MoonShadowAI 文本生成器（MoBu 二进制版）")
    print("输入 exit 退出\n")

    while True:
        word = input("起始词：").strip()
        if word == "exit":
            break
        if not word:
            continue

        sentence = generate(word)
        print(sentence)
