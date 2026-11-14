# -*- coding: utf-8 -*-
"""配置加载：支持点号取值/赋值，缺省时写出默认 train.yaml。"""
import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CONFIG = {
    "model": {"embed_dim": 80, "hidden_dim": 160, "num_layers": 1, "dropout": 0.2, "block_size": 64},
    "train": {"batch_size": 256, "epochs": 50, "learning_rate": 0.005, "weight_decay": 1e-5,
              "grad_clip": 1.0, "early_stop_patience": 5, "min_delta": 1e-4},
    "data": {"seq_path": "neural/data/sequences.mobu", "vocab_path": "neural/data/vocab.mobu",
             "train_split": 0.9},
    "runtime": {"threads": 4, "interop_threads": 1},
    "autotune": {"enabled": True},
    "checkpoint": {"dir": "neural/checkpoints", "best_name": "best_model.pt", "save_every": 5},
}


def _deep_copy(d):
    if isinstance(d, dict):
        return {k: _deep_copy(v) for k, v in d.items()}
    return d


def _merge(base, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v


class Config:
    def __init__(self, data=None, path=None):
        self._data = _deep_copy(DEFAULT_CONFIG if data is None else data)
        self.path = path

    def get(self, dotted, default=None):
        cur = self._data
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def set(self, dotted, value):
        parts = dotted.split(".")
        cur = self._data
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    def as_dict(self):
        return _deep_copy(self._data)

    def save(self, path=None):
        p = path or self.path or os.path.join(PROJECT_ROOT, "config", "train.yaml")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._data, f, allow_unicode=True, sort_keys=False)
        return p


def load_config(path=None):
    """读 yaml；path 为 None 时依次尝试 config/train.yaml、neural/train.yaml，都没有则写默认。"""
    candidates = [path] if path else [
        os.path.join(PROJECT_ROOT, "config", "train.yaml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.yaml"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            cfg = Config(_deep_copy(DEFAULT_CONFIG), path=p)
            _merge(cfg._data, data)
            return cfg
    cfg = Config(path=candidates[0])
    cfg.save()
    return cfg
