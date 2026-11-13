import os
import jieba
import json
from collections import Counter, defaultdict
import re

punctuation_pattern = re.compile(r'^[，。！？、；：""''（）【】《》\s]+$')


def read_material(file_path):
    """读取单个文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text):
    """去除多余空白，保留中文、英文、数字和常见标点"""
    import re
    text = re.sub(r'\s+', '', text)
    return text


def build_vocab_and_bigram(materials_dir):
    """读取 materials 下所有文件，分词、统计词频和 bigram"""
    vocab_counter = Counter()
    bigram_counter = defaultdict(lambda: defaultdict(int))

    for filename in os.listdir(materials_dir):
        filepath = os.path.join(materials_dir, filename)
        if not os.path.isfile(filepath):
            continue
        text = read_material(filepath)
        text = clean_text(text)
        tokens = list(jieba.cut(text))

        # 把标点粘到前一个词上
        merged_tokens = []
        for token in tokens:
            if punctuation_pattern.match(token) and merged_tokens:
                merged_tokens[-1] += token
            else:
                merged_tokens.append(token)

        # 统计词频
        for token in merged_tokens:
            vocab_counter[token] += 1

        # 统计 bigram
        for i in range(len(merged_tokens) - 1):
            bigram_counter[merged_tokens[i]][merged_tokens[i + 1]] += 1

    print(f"清洗前：{len(vocab_counter)} 个词")

    # 清洗：去长度小于1的词、去纯标点
    cleaned_vocab = {}
    for word, count in vocab_counter.items():
        if len(word) < 1:
            continue
        if punctuation_pattern.match(word):
            continue
        cleaned_vocab[word] = count

    print(f"清洗后：{len(cleaned_vocab)} 个词")

    # 清洗 bigram：只保留 key 和 value 都在 cleaned_vocab 中的条目
    cleaned_bigram = {}
    for first, followers in bigram_counter.items():
        if first not in cleaned_vocab:
            continue
        filtered_followers = {
            second: count
            for second, count in followers.items()
            if second in cleaned_vocab
        }
        if filtered_followers:
            cleaned_bigram[first] = filtered_followers

    print(f"bigram 条目数：{len(cleaned_bigram)}")

    return cleaned_vocab, cleaned_bigram


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    materials_dir = os.path.join(script_dir, "materials")
    vocab, bigram = build_vocab_and_bigram(materials_dir)
    print(f"词表大小：{len(vocab)}")
    print(f"bigram 条目数：{len(bigram)}")
