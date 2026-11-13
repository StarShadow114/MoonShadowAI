import os
import re
import struct
from read_book import build_vocab_and_bigram

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
models_dir = os.path.join(root_dir, "models")

os.makedirs(models_dir, exist_ok=True)


def save_vocab_binary(vocab, filepath):
    """将词表写入二进制文件：每条记录 = 编号(4B) + 内容长度(2B) + UTF-8内容"""
    with open(filepath, "wb") as f:
        for idx, word in enumerate(sorted(vocab), start=1):
            encoded = word.encode("utf-8")
            f.write(struct.pack("<I", idx))        # 编号，4字节
            f.write(struct.pack("<H", len(encoded))) # 内容长度，2字节
            f.write(encoded)                         # UTF-8内容


def save_bigram_binary(bigram, filepath):
    """将 bigram 写入二进制文件：每条记录 = 源编号(4B) + 目标编号(4B) + 权重(1B) + 分数(1B)"""
    # 先构建词→编号映射
    all_words = set(bigram.keys())
    for followers in bigram.values():
        all_words.update(followers.keys())
    word_to_id = {word: idx + 1 for idx, word in enumerate(sorted(all_words))}

    # 写入词表
    vocab_path = os.path.join(os.path.dirname(filepath), "vocab.mobu")
    save_vocab_binary(all_words, vocab_path)

    # 写入 bigram
    with open(filepath, "wb") as f:
        for first, followers in bigram.items():
            first_id = word_to_id[first]
            for second, count in followers.items():
                second_id = word_to_id[second]
                weight = min(count, 255)
                score = 0
                f.write(struct.pack("<IIBB", first_id, second_id, weight, score))

    print(f"词表已保存：{vocab_path}")
    print(f"Bigram 已保存：{filepath}")


if __name__ == "__main__":
    materials_dir = os.path.join(root_dir, "training", "materials")
    vocab, bigram = build_vocab_and_bigram(materials_dir)

    bigram_path = os.path.join(models_dir, "bigram.mobu")
    save_bigram_binary(bigram, bigram_path)
