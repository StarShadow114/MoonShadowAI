# -*- coding: utf-8 -*-
"""CPU 基准：实测 step/s 并挑一档 batch_size 预设。"""
import gc
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .model import LSTMLanguageModel  # noqa: E402

PRESETS = [("slow", 128), ("medium", 256), ("fast", 512)]
BENCH_STEPS = 30
WARMUP_STEPS = 3


def _release(*models):
    for m in models:
        try:
            del m
        except Exception:
            pass
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def autotune(cfg, vocab_size, device, steps=BENCH_STEPS, warmup=WARMUP_STEPS):
    """跑随机 batch 的 forward+backward 计时，返回 (档位名, step/s)，并把 batch_size 写回 cfg。"""
    device = torch.device(device)
    block_size = int(cfg.get("model.block_size", 64))
    embed_dim = int(cfg.get("model.embed_dim", 80))
    hidden_dim = int(cfg.get("model.hidden_dim", 160))
    num_layers = int(cfg.get("model.num_layers", 1))
    dropout = float(cfg.get("model.dropout", 0.2))
    batch_size = max(1, int(cfg.get("train.batch_size", 256)))

    try:
        model = LSTMLanguageModel(vocab_size=vocab_size, embed_dim=embed_dim,
                                  hidden_dim=hidden_dim, num_layers=num_layers,
                                  dropout=dropout, block_size=block_size).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        def one_step():
            x = torch.randint(0, max(2, vocab_size), (batch_size, block_size), device=device)
            y = torch.randint(0, max(2, vocab_size), (batch_size, block_size), device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, _h = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for _ in range(max(0, warmup)):
            one_step()
        t0 = time.perf_counter()
        for _ in range(max(1, steps)):
            one_step()
        elapsed = time.perf_counter() - t0
        step_per_sec = steps / elapsed if elapsed > 0 else 0.0
        _release(model, optimizer, criterion)
    except Exception:
        _release()
        return "fallback", 0.0

    if step_per_sec >= 12.0:
        name, chosen = PRESETS[2]
    elif step_per_sec >= 4.0:
        name, chosen = PRESETS[1]
    else:
        name, chosen = PRESETS[0]
    if chosen != batch_size:
        cfg.set("train.batch_size", chosen)
    return name, step_per_sec


__all__ = ["autotune", "PRESETS"]
