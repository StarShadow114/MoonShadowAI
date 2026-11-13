"""
MoonShadowAI - 自动训练脚本
"""
import os
import sys
import random
import struct
import time

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

# 配置
MAX_ROUNDS = 20000

# 中文黑名单
BLACKLIST = {
    "无法", "错误", "失败", "异常", "崩溃",
    "无效", "空值", "未知", "出错", "报错",
    "暂无", "无内容", "未定义", "请稍后", "网络繁忙"
}


# ---------- 工具函数 ----------

def load_vocab_binary(filepath):
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
            bigram[first][second] = {"weight": weight, "pos_count": 0, "neg_count": 0}
    return bigram


def save_bigram_binary(bigram, filepath, word_to_id):
    with open(filepath, "wb") as f:
        for first, followers in bigram.items():
            first_id = word_to_id.get(first)
            if first_id is None:
                continue
            for second, data in followers.items():
                second_id = word_to_id.get(second)
                if second_id is None:
                    continue
                weight = data["weight"]
                score = 0
                f.write(struct.pack("<IIBB", first_id, second_id, weight, score))


def generate_with_path(start_word, bigram, vocab, max_length=15):
    if start_word not in vocab:
        return "", []
    if start_word not in bigram:
        return start_word, []

    result = [start_word]
    current = start_word
    path = []

    for _ in range(max_length - 1):
        if current not in bigram:
            break
        next_words = bigram[current]
        if not next_words:
            break

        next_word = pick_next({k: v["weight"] for k, v in next_words.items()})
        path.append((current, next_word))
        result.append(next_word)
        current = next_word

    return "".join(result) + "。", path


def get_unique_word_ratio(sentence):
    """计算句子中不同字的占比"""
    words = sentence.replace("。", "").replace("，", "").replace("、", "").replace("！", "").replace("？", "").replace("，", "")
    if not words:
        return 0
    unique = len(set(words))
    total = len(words)
    return unique / total


def has_repeated_pattern(sentence):
    """检测是否有连续重复的字（≥3次）"""
    words = list(sentence.replace("。", "").replace("，", ""))
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2]:
            return True
    return False


def auto_score(sentence):
    """
    自动评分：好评门槛收紧
    +1 好：长度8~30 + 无黑名单 + 无重复 + 不同字占比≥40%
    -1 差：长度<5或>80 + 含黑名单 + 重复
     0 不确定：其余
    """
    length = len(sentence)

    # 差评条件
    if length < 5 or length > 80:
        return -1
    for word in BLACKLIST:
        if word in sentence:
            return -1
    if has_repeated_pattern(sentence):
        return -1
    if "的的" in sentence:
        return -1

    # 好评条件（收紧）
    if 8 <= length <= 50 and get_unique_word_ratio(sentence) >= 0.35:
        return +1

    # 不确定
    return 0


def adjust_bigram_weight(bigram, path, delta):
    changed = 0
    for first, second in path:
        data = bigram[first][second]
        old_weight = data["weight"]
        new_weight = old_weight + delta
        if new_weight < 1:
            new_weight = 1
        if new_weight > 240:
            new_weight = 240
        bigram[first][second]["weight"] = new_weight

        if delta > 0:
            bigram[first][second]["pos_count"] += 1
        elif delta < 0:
            bigram[first][second]["neg_count"] += 1

        if new_weight != old_weight:
            changed += 1
    return changed


def progress_bar(current, total, bar_length=40):
    filled = int(bar_length * current // total)
    bar = "█" * filled + "░" * (bar_length - filled)
    pct = current / total * 100
    return f"[{bar}] {pct:.1f}% ({current}/{total})"


# ---------- 主流程 ----------

if __name__ == "__main__":
    vocab_path = os.path.join(models_dir, "vocab.mobu")
    bigram_path = os.path.join(models_dir, "bigram.mobu")

    id_to_word, word_to_id = load_vocab_binary(vocab_path)
    vocab = set(id_to_word.values())
    bigram = load_bigram_binary(bigram_path, id_to_word)

    available_starts = list(vocab)
    uncertain_samples = []
    stats = {"good": 0, "bad": 0, "uncertain": 0}

    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{YELLOW}   MoonShadowAI 自动训练模式{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"   训练轮次：{MAX_ROUNDS}")
    print(f"   词表大小：{len(vocab)}")
    print(f"   可起始词数：{len(available_starts)}")
    print(f"{BOLD}{'='*60}{RESET}")

    round_count = 0
    start_time = time.time()

    while round_count < MAX_ROUNDS and available_starts:
        start_word = random.choice(available_starts)
        sentence, path = generate_with_path(start_word, bigram, vocab)

        # 句子太短（不含标点少于5个字），直接差评
        clean_len = len(sentence.replace("。", "").replace("，", "").replace(" ", ""))
        if not path or clean_len < 5:
            score = -1
        else:
            score = auto_score(sentence)

        if score == +1:
            adjust_bigram_weight(bigram, path, +1)
            stats["good"] += 1
            round_count += 1
        elif score == -1:
            # 如果是冷门词导致句子太短，也要更新权重
            if path:
                adjust_bigram_weight(bigram, path, -1)
            stats["bad"] += 1
            round_count += 1
        else:
            uncertain_samples.append({
                "start_word": start_word,
                "sentence": sentence,
                "path": path
            })
            stats["uncertain"] += 1
            continue

        elapsed = time.time() - start_time
        eta = (elapsed / round_count) * (MAX_ROUNDS - round_count) if round_count > 0 else 0
        bar = progress_bar(round_count, MAX_ROUNDS)
        good_color = GREEN if stats["good"] > stats["bad"] else RED
        bad_color = RED if stats["bad"] > 0 else GREEN
        print(f"\r{bar}  {good_color}好:{stats['good']}{RESET} {bad_color}差:{stats['bad']}{RESET} {YELLOW}跳:{stats['uncertain']}{RESET}  剩余:{eta:.0f}s", end="")

    print(f"\n\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{GREEN}训练完成！共 {stats['good'] + stats['bad']} 轮有效训练{RESET}")
    print(f"不确定样本总数：{len(uncertain_samples)}")

    if uncertain_samples:
        sample_size = min(10, len(uncertain_samples))
        review_sample = random.sample(uncertain_samples, sample_size)
        print(f"\n{BOLD}{YELLOW}随机抽取 {sample_size} 个样本供人工复核：{RESET}")
        for i, sample in enumerate(review_sample, 1):
            print(f"\n[{i}] {CYAN}起始词：{sample['start_word']}{RESET}")
            print(f"    {sample['sentence']}")
            while True:
                cmd = input(f"   打分？（{GREEN}p{RESET}加分/{RED}m{RESET}扣分/{YELLOW}回车{RESET}跳过）：").strip().lower()
                if cmd == "p":
                    if sample["path"]:
                        adjust_bigram_weight(bigram, sample["path"], +1)
                        stats["good"] += 1
                    print(f"   {GREEN}已加分{RESET}")
                    break
                elif cmd == "m":
                    if sample["path"]:
                        adjust_bigram_weight(bigram, sample["path"], -1)
                        stats["bad"] += 1
                    print(f"   {RED}已扣分{RESET}")
                    break
                elif cmd == "":
                    print(f"   {YELLOW}已跳过{RESET}")
                    break
                else:
                    print(f"   输入无效，请输入 {GREEN}p{RESET}/{RED}m{RESET} 或直接回车")

        save_bigram_binary(bigram, bigram_path, word_to_id)
        print(f"\n{GREEN}人工复核结果已保存至模型{RESET}")

        skipped = len(uncertain_samples) - sample_size
        if skipped > 0:
            print(f"{YELLOW}其余 {skipped} 个不确定样本已直接跳过丢弃{RESET}")

    save_bigram_binary(bigram, bigram_path, word_to_id)
    print(f"\n{GREEN}模型已保存：{bigram_path}{RESET}")

    elapsed_total = time.time() - start_time
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}训练统计{RESET}")
    print(f"{'='*60}")
    print(f"   有效训练轮次：{stats['good'] + stats['bad']}")
    print(f"   {GREEN}好评：{stats['good']}{RESET}")
    print(f"   {RED}差评：{stats['bad']}{RESET}")
    print(f"   {YELLOW}不确定跳过：{stats['uncertain']}{RESET}")
    print(f"   人工复核：{min(10, len(uncertain_samples))} 个")
    print(f"   总耗时：{elapsed_total:.1f} 秒")
    print(f"{BOLD}{'='*60}{RESET}")
