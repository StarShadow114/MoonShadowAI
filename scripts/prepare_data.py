# -*- coding: utf-8 -*-
"""prepare_data：语料 -> neural/data/vocab.mobu + neural/data/sequences.mobu

用法：
    python scripts/prepare_data.py                # 用内置示例语料
    python scripts/prepare_data.py 语料.txt       # 指定 UTF-8 语料
    python scripts/prepare_data.py 语料.txt 128   # 指定 block_size

字符级分词，特殊 token：0=PAD 1=UNK 2=BOS 3=EOS，其后按字频降序入表。
脚本幂等：重复运行产出同样结构的文件（同语料下词表与序列完全一致）。
"""
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "neural")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mobutools import MobuModel  # noqa: E402
from neural.config import load_config  # noqa: E402

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

SAMPLE_CORPUS = """人工智能正在改变人类理解世界的方式。深度学习的出现，让机器从数据中自学规律成为可能。
神经网络模仿人脑的神经元连接，一层一层地把原始信号抽象成高层概念。自然语言处理是其中最有温度的分支，
它让计算机读懂文字、听懂语言，也能写出通顺连贯的句子。早期的语言模型依赖统计方法，用 n-gram 猜测下一个词；
后来循环神经网络登场，隐状态像一条记忆的河流，把前文的信息一路带到远方。长短时记忆网络 LSTM 用门控机制
缓解了梯度消失：遗忘门决定丢弃什么，输入门决定记住什么，输出门决定呈现什么，三个门协同工作，让长距离依赖
不再遥不可及。训练时用随时间反向传播，梯度沿着时间轴一步步回传，CPU 上跑得慢，却每一步都看得见。
虽然今天 Transformer 凭借注意力机制占领了主流舞台，LSTM 依然轻量、稳定、好调试，在纯 CPU 的手机环境里
也能踏实训练，是理解语言建模最好的入门路径。训练语言模型的核心目标其实很简单：给定前面的 token，
预测下一个 token 的概率分布。交叉熵衡量预测与真实之间的差距，困惑度是交叉熵的指数，数值越低说明模型越不意外。
数据准备阶段要做三件事：分词、建词表、切块。中文按字符切分足够好用，每个汉字与常见标点都成为一个 token。
词表按字频降序排列，高频字拿到小 id，低频字排在后面，未知字统一映射到 UNK。切块时把整篇语料编码成一维 id 序列，
再按 block_size 滑窗截取，每块额外多取一个 token 作为监督目标，于是输入 x 与目标 y 天然错开一位。
MoonShadowAI 就是这样从零起步的：不用任何高层训练框架，只用 PyTorch 手写 LSTM，用 Mobu 二进制容器保存词表与权重，
在纯 CPU 的手机上慢慢开智。慢一点没关系，重要的是每一行代码都亲手写过，每一次损失下降都亲眼见过。
当困惑度从两位数跌到个位数，模型开始吐出像模像样的句子，那一刻你会明白，所谓智能，不过是大量微小正确的累积。
"""


def build_vocab(text, min_freq=1):
    """特殊 token 占位后，按字频降序排列；同频按字符码位排，保证结果确定可复现。"""
    counter = Counter(ch for ch in text if not ch.isspace())
    items = [(ch, c) for ch, c in counter.items() if c >= min_freq]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    vocab = list(SPECIAL_TOKENS) + [ch for ch, _c in items]
    return vocab, counter


def encode(text, token_to_id):
    unk = token_to_id["<UNK>"]
    return [token_to_id.get(ch, unk) for ch in text if not ch.isspace()]


def slice_blocks(ids, block_size):
    """滑窗切块，每块多取 1 个 token 作为 y 的末位目标。"""
    usable = len(ids) - 1
    if usable <= 0:
        raise ValueError("语料太短，编码后只有 %d 个有效 token，无法切块" % usable)
    rows = []
    for start in range(0, usable, block_size):
        chunk = ids[start:start + block_size + 1]
        if len(chunk) < block_size + 1:
            chunk = chunk + [0] * (block_size + 1 - len(chunk))   # PAD 补齐尾块
        rows.append(chunk)
    return rows


def write_vocab_mobu(path, vocab, source, total_tokens):
    mobu = MobuModel()
    mobu.set_vocab(vocab)
    mobu.set_meta("source", source)
    mobu.set_meta("token_count", total_tokens)
    mobu.set_meta("special_tokens", " ".join("%d=%s" % (i, t) for i, t in enumerate(SPECIAL_TOKENS)))
    mobu.set_meta("tokenizer", "char")
    mobu.set_meta("encoding", "utf-8")
    mobu.save(path)
    return mobu


def write_sequences_mobu(path, rows, block_size, vocab_size):
    import numpy as np
    mobu = MobuModel()
    arr = np.asarray(rows, dtype=np.int64)
    mobu.add_weight(name="tokens", tensor_bytes=arr.tobytes(),
                    dtype_code=0x04, shape=list(arr.shape))
    mobu.set_meta("block_size", block_size)
    mobu.set_meta("seq_len", block_size + 1)
    mobu.set_meta("num_blocks", arr.shape[0])
    mobu.set_meta("vocab_size", vocab_size)
    mobu.set_meta("dtype", "int64")
    mobu.save(path)
    return mobu


def main():
    cfg = load_config()
    block_size = int(cfg.get("model.block_size", 64))
    vocab_path = cfg.get("data.vocab_path", "neural/data/vocab.mobu")
    seq_path = cfg.get("data.seq_path", "neural/data/sequences.mobu")

    src = sys.argv[1] if len(sys.argv) > 1 else None
    if len(sys.argv) > 2:
        try:
            block_size = int(sys.argv[2])
        except ValueError:
            print("[warn] block_size 参数非法，沿用配置值 %d" % block_size)

    if src:
        src = os.path.expanduser(src)
        if not os.path.isabs(src):
            src = os.path.join(PROJECT_ROOT, src)
        if not os.path.exists(src):
            print("[错误] 语料文件不存在：%s" % src)
            return 1
        with open(src, "r", encoding="utf-8") as f:
            text = f.read()
        source_label = os.path.basename(src)
    else:
        text = SAMPLE_CORPUS
        source_label = "builtin_sample_corpus"

    vocab_path = _abs(vocab_path)
    seq_path = _abs(seq_path)

    vocab, counter = build_vocab(text)
    token_to_id = {t: i for i, t in enumerate(vocab)}
    ids = encode(text, token_to_id)
    rows = slice_blocks(ids, block_size)
    total_tokens = len(ids)

    write_vocab_mobu(vocab_path, vocab, source_label, total_tokens)
    write_sequences_mobu(seq_path, rows, block_size, len(vocab))

    # ---- 回读校验 ----
    v = MobuModel.load(vocab_path)
    s = MobuModel.load(seq_path)
    assert v.vocab_size() == len(vocab), "词表回读数量不符"
    assert s.get_weight("tokens") is not None, "序列回读缺 tokens 段"
    blk = s.get_weight("tokens")
    assert list(blk["shape"]) == [len(rows), block_size + 1], "序列形状不符"

    print("语料来源   : %s" % source_label)
    print("字符总量   : %d（去空白后 %d）" % (len(text), total_tokens))
    print("词表大小   : %d（特殊 %d + 普通 %d）" % (v.vocab_size(), len(SPECIAL_TOKENS),
                                              v.vocab_size() - len(SPECIAL_TOKENS)))
    print("总 token 数: %d" % total_tokens)
    print("块数       : %d（block_size=%d，块宽 %d）" % (len(rows), block_size, block_size + 1))
    print("词表前 20  : %s" % " ".join("%s" % t for t in v.get_vocab()[:20]))
    print("高频 top10 : %s" % " ".join("%s×%d" % (c, n) for c, n in counter.most_common(10)))
    print("产物       : %s" % vocab_path)
    print("             %s" % seq_path)
    print("校验       : 词表 %d 词 OK / 权重 tokens %s int64 OK"
          % (v.vocab_size(), "x".join(str(d) for d in blk["shape"])))
    return 0


def _abs(p):
    p = os.path.expanduser(str(p))
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


if __name__ == "__main__":
    sys.exit(main())
