# -*- coding: utf-8 -*-
"""从 sequences.mobu 构建 DataLoader（CPU / 单进程，适配 Termux proot）。"""
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mobutools import MobuModel, mobu_to_state_dict  # noqa: E402


def load_tokens(seq_path):
    """读取 sequences.mobu 里的 "tokens" 权重，返回 LongTensor (N, block_size+1)。"""
    if not os.path.exists(seq_path):
        raise FileNotFoundError(
            "找不到序列文件 %s，请先运行：python scripts/prepare_data.py" % seq_path)
    mobu = MobuModel.load(seq_path)
    blk = mobu.get_weight("tokens")
    if blk is None:
        raise KeyError("%s 里没有名为 tokens 的权重段" % seq_path)
    sd = mobu_to_state_dict(mobu)
    tensor = sd["tokens"].long()
    expect = 1
    for d in blk["shape"]:
        expect *= d
    if tensor.numel() != expect:
        raise ValueError("tokens 元素数不符: %d != %d" % (tensor.numel(), expect))
    if tensor.dim() != 2:
        raise ValueError("tokens 应为二维 (num_blocks, block_size+1)，实为 %s" % (list(tensor.shape),))
    return tensor


def create_dataloader(seq_path, block_size, batch_size, shuffle=True, split=0.9,
                      split_idx=0, seed=1337):
    """切出训练/验证 DataLoader。split_idx=0 训练段，1 验证段。"""
    tokens = load_tokens(seq_path)
    if tokens.size(1) != block_size + 1:
        raise ValueError(
            "序列文件块宽 %d 与配置 block_size=%d 不匹配，请重跑 prepare_data.py"
            % (tokens.size(1), block_size))

    n = tokens.size(0)
    split = min(1.0, max(0.0, float(split)))
    cut = int(n * split)
    cut = max(1, min(cut, n - 1)) if n > 1 else n

    if split_idx == 0:
        chunk = tokens[:cut]
    elif split_idx == 1:
        chunk = tokens[cut:]
        if chunk.size(0) == 0:
            chunk = tokens[:1]
    else:
        raise ValueError("split_idx 只能是 0（训练）或 1（验证）")

    x = chunk[:, :-1].contiguous()
    y = chunk[:, 1:].contiguous()
    dataset = TensorDataset(x, y)

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed + int(split_idx))
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
        return DataLoader(dataset, batch_size=int(batch_size), sampler=sampler,
                          num_workers=0, pin_memory=False, drop_last=False)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=False,
                      num_workers=0, pin_memory=False, drop_last=False)


__all__ = ["load_tokens", "create_dataloader"]
