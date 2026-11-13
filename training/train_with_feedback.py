"""
MoonShadowAI - 人工反馈训练脚本（MoBu 二进制版）
"""
import os
import sys
import random
import struct

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
models_dir = os.path.join(root_dir, "models")

sys.path.insert(0, os.path.join(root_dir, "AISRC"))
from sentence_generator import pick_next

# ANSI 颜色
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


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


def load_bigram_binary(filepath, id_to_word):
    """加载 bigram，返回 bigram_dict"""
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


def save_bigram_binary(bigram, filepath, word_to_id):
    """把调整后的 bigram 写回文件"""
    with open(filepath, "wb") as f:
        for first, followers in bigram.items():
            first_id = word_to_id.get(first)
            if first_id is None:
                continue
            for second, weight in followers.items():
                second_id = word_to_id.get(second)
                if second_id is None:
                    continue
                score = 0
                f.write(struct.pack("<IIBB", first_id, second_id, weight, score))


def generate_with_path(start_word, bigram, vocab, max_length=20):
    """生成句子并记录路径"""
    if start_word not in vocab:
        return f"'{start_word}' 不在词表里", []
    if start_word not in bigram:
        return f"'{start_word}' 后面没有接任何词", []

    result = [start_word]
    current = start_word
    path = []

    for _ in range(max_length - 1):
        if current not in bigram:
            break
        next_words = bigram[current]
        if not next_words:
            break

        next_word = pick_next(next_words)
        path.append((current, next_word))
        result.append(next_word)
        current = next_word

    return "".join(result) + "。", path


def adjust_bigram(bigram, path, delta):
    """沿路径调整权重"""
    changed = 0
    for first, second in path:
        old = bigram[first][second]
        new = old + delta
        if new < 1:
            new = 1
        if new > 255:
            new = 255
        bigram[first][second] = new
        if new != old:
            changed += 1
    return changed


def show_path(path):
    """打印路径"""
    if not path:
        return
    steps = [f"{a} → {b}" for a, b in path]
    print(f"{CYAN}路径：{'  '.join(steps)}{RESET}")


if __name__ == "__main__":
    vocab_path = os.path.join(models_dir, "vocab.mobu")
    bigram_path = os.path.join(models_dir, "bigram.mobu")

    id_to_word, word_to_id = load_vocab_binary(vocab_path)
    vocab = set(id_to_word.values())
    bigram = load_bigram_binary(bigram_path, id_to_word)

    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{YELLOW}   MoonShadowAI 训练模式（MoBu 二进制版）{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f" {GREEN}p{RESET} = plus（加分）")
    print(f" {RED}m{RESET} = minus（减分）")
    print(f" 直接回车 = 跳过")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{RED}{BOLD}⚠ 你的评价直接影响模型的最终智力。认真对待每一次判断。{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    stats = {"plus": 0, "minus": 0, "skip": 0}
    available_starts = [w for w in vocab if w in bigram]

    while True:
        if not available_starts:
            print(f"{RED}没有可用起始词，无法继续训练。{RESET}")
            break

        start_word = random.choice(available_starts)

        sentence, path = generate_with_path(start_word, bigram, vocab)
        print(f"\n{BOLD}起始词：{CYAN}{start_word}{RESET}")
        print(f"{GREEN}→{RESET} {sentence}")

        if not path:
            print(f"{YELLOW}（这条没有路径，跳过打分）{RESET}")
            continue

        show_path(path)

        score = input(f"打分？（{GREEN}p{RESET}/{RED}m{RESET}/回车跳过）：").strip().lower()
        if score == "p":
            n = adjust_bigram(bigram, path, +1)
            save_bigram_binary(bigram, bigram_path, word_to_id)
            stats["plus"] += 1
            print(f"{GREEN}✓ 已加分，{n} 条边权重上调{RESET}")
        elif score == "m":
            n = adjust_bigram(bigram, path, -1)
            save_bigram_binary(bigram, bigram_path, word_to_id)
            stats["minus"] += 1
            print(f"{RED}✗ 已减分，{n} 条边权重下调{RESET}")
        else:
            stats["skip"] += 1
            print(f"{YELLOW}— 跳过{RESET}")

    print(f"\n{BOLD}训练结束{RESET}")
    print(f"{GREEN}加分 {stats['plus']} 次{RESET} / {RED}减分 {stats['minus']} 次{RESET} / {YELLOW}跳过 {stats['skip']} 次{RESET}")
