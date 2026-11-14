# -*- coding: utf-8 -*-
"""generate：加载 checkpoint 自回归采样生成中文文本。

用法：
    python scripts/generate.py -n 2
    python scripts/generate.py -p "深度学习" -n 3 --temp 0.8 --topk 20 --topp 0.9 --len 100
"""
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "neural")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from mobutools import MobuModel  # noqa: E402
from neural.model import LSTMLanguageModel, count_params  # noqa: E402
from neural.dashboard import C  # noqa: E402


def _abs(p):
    p = str(p or "")
    if not p:
        return p
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


class VocabCodec:
    """词表编解码，接口名兼容探测：encode/decode、token_to_id/id_to_word、list。"""

    def __init__(self, mobu):
        self.mobu = mobu
        self.vocab = list(mobu.get_vocab()) if hasattr(mobu, "get_vocab") else []
        if not self.vocab:
            raise RuntimeError("词表为空：请确认 vocab 文件正确（python scripts/prepare_data.py）")
        self._table = {t: i for i, t in enumerate(self.vocab)}
        self.enc_name = self._probe(("encode", "token_to_id", "word_to_id", "str_to_id"))
        self.dec_name = self._probe(("decode", "id_to_word", "token_from_id", "id_to_token"))

    def _probe(self, names):
        for n in names:
            fn = getattr(self.mobu, n, None)
            if callable(fn):
                return n
        return None

    def describe(self):
        return "encode=%s decode=%s" % (self.enc_name or "fallback(逐字查表)",
                                       self.dec_name or "fallback(列表索引)")

    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        if self.enc_name:
            fn = getattr(self.mobu, self.enc_name)
            try:
                out = fn(text)
                if isinstance(out, (list, tuple)):
                    return [int(i) for i in out]
                if isinstance(out, int):
                    return [out]
            except Exception:
                pass
        return [self._table.get(ch, 1) for ch in str(text)]

    def decode(self, ids):
        ids = [int(i) for i in ids]
        if self.dec_name:
            fn = getattr(self.mobu, self.dec_name)
            try:
                out = fn(ids)
                if isinstance(out, str):
                    return out
                if isinstance(out, (list, tuple)):
                    return "".join(str(x) for x in out)
            except Exception:
                pass
        return "".join(self.vocab[i] if 0 <= i < len(self.vocab) else "" for i in ids)


def load_ckpt(ckpt_path):
    path = _abs(ckpt_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "找不到 checkpoint：%s\n先跑一次训练：python scripts/train.py -E train.epochs=2" % path)
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError("checkpoint 格式异常，期望 dict，实为 %s" % type(obj).__name__)
    if "model_state_dict" in obj:
        sd, meta = obj["model_state_dict"], obj
    else:
        sd, meta = obj, {}
    required = ("epoch", "vocab_size", "embed_dim", "hidden_dim", "num_layers", "block_size")
    missing = [k for k in required if k not in meta]
    if missing:
        raise KeyError(
            "checkpoint 缺少架构信息 %s。请用 scripts/train.py 产出的 best_model.pt。" % ", ".join(missing))
    return sd, meta


def build_model(sd, meta, device):
    model = LSTMLanguageModel(
        vocab_size=int(meta["vocab_size"]),
        embed_dim=int(meta["embed_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
        num_layers=int(meta["num_layers"]),
        dropout=0.2,
        block_size=int(meta["block_size"]),
    ).to(device)
    res = model.load_state_dict(sd, strict=True)
    missing, unexpected = (res.missing_keys, res.unexpected_keys) if hasattr(res, "missing_keys") else (res, [])
    if missing or unexpected:
        raise RuntimeError("strict 载入不一致：missing=%s unexpected=%s" % (missing, unexpected))
    model.eval()
    return model


@torch.no_grad()
def sample(model, codec, prompt, max_len=100, temperature=0.8, top_k=20, top_p=0.9,
           device=None, seed=None):
    device = torch.device(device or model.device)
    if seed is not None:
        torch.manual_seed(seed)
    temp = max(1e-5, float(temperature))
    block_size = int(getattr(model, "block_size", 64))

    ids = [i for i in codec.encode(prompt or "") if i >= 4]
    if not ids:
        pool = [i for i in range(4, min(codec.vocab_size(), 400))]
        ids = [pool[torch.randint(0, len(pool), (1,)).item()] if pool else 4]

    ctx = list(ids)
    hidden = None
    if len(ctx) > 1:
        head = ctx[:-1][-block_size:]
        logits, hidden = model(torch.tensor([head], dtype=torch.long, device=device), None)
    generated = []
    for _ in range(max(1, int(max_len))):
        window = ctx[-block_size:]
        logits, hidden = model(torch.tensor([window], dtype=torch.long, device=device), hidden)
        log = logits[0, -1].float() / temp
        k = int(top_k or 0)
        if 0 < k < log.numel():
            vals = torch.topk(log, k).values[-1]
            log = torch.where(log >= vals, log, torch.full_like(log, float("-inf")))
        p = float(top_p or 1.0)
        if 0 < p < 1.0:
            order = torch.argsort(log, descending=True)
            sl = log[order]
            cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
            mask = cum > p
            if bool(mask.all().item()):
                mask[0] = False
            sl = torch.where(mask, torch.full_like(sl, float("-inf")), sl)
            out = torch.full_like(log, float("-inf"))
            out[order] = sl
            log = out
        if not torch.isfinite(log).any().item():
            break
        nxt = int(torch.multinomial(torch.softmax(log, dim=-1), 1).item())
        if nxt <= 3 or nxt >= codec.vocab_size():
            break
        generated.append(nxt)
        ctx.append(nxt)
    return codec.decode(generated)


COLORS = [C.CYAN, C.GREEN, C.MAGENTA, C.YELLOW, C.BLUE]


def main():
    ap = argparse.ArgumentParser(
        description="MoonShadowAI 文本生成（纯 CPU，温度 + top-k + top-p）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="示例：python scripts/generate.py -p \"深度学习\" -n 3 --temp 0.8 --topk 20 --topp 0.9 --len 100")
    ap.add_argument("--prompt", "-p", default="", help="提示词，空则从词表随机起头")
    ap.add_argument("--ckpt", default="neural/checkpoints/best_model.pt", help="checkpoint 路径")
    ap.add_argument("--vocab", default="neural/data/vocab.mobu", help="词表 .mobu 路径")
    ap.add_argument("--n", "-n", type=int, default=5, dest="n", help="生成条数")
    ap.add_argument("--temp", type=float, default=0.8, help="采样温度（下限 1e-5）")
    ap.add_argument("--topk", type=int, default=20, help="top-k，0 表示关闭")
    ap.add_argument("--topp", type=float, default=0.9, help="top-p 核采样，1.0 表示关闭")
    ap.add_argument("--len", type=int, default=100, help="每条最大新生成 token 数")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    ap.add_argument("--quiet", action="store_true", help="只输出文本，不带颜色与头部信息")
    args = ap.parse_args()

    device = torch.device("cpu")
    if hasattr(torch, "set_num_threads"):
        try:
            import yaml
            cfg_path = os.path.join(PROJECT_ROOT, "config", "train.yaml")
            threads = 4
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    threads = int((yaml.safe_load(f) or {}).get("runtime", {}).get("threads", 4))
            torch.set_num_threads(max(1, threads))
        except Exception:
            pass

    vocab_path = _abs(args.vocab)
    if not os.path.exists(vocab_path):
        print("%s[错误] 找不到词表 %s，先跑 python scripts/prepare_data.py%s" % (C.RED, vocab_path, C.RESET))
        return 1
    try:
        sd, meta = load_ckpt(args.ckpt)
    except (FileNotFoundError, KeyError, TypeError, RuntimeError) as exc:
        print("%s[错误] %s%s" % (C.RED, exc, C.RESET))
        return 1
    try:
        model = build_model(sd, meta, device)
    except RuntimeError as exc:
        print("%s[错误] 模型重建/载入失败：%s%s" % (C.RED, exc, C.RESET))
        return 1

    mobu = MobuModel.load(vocab_path)
    try:
        codec = VocabCodec(mobu)
    except RuntimeError as exc:
        print("%s[错误] %s%s" % (C.RED, exc, C.RESET))
        return 1

    if not args.quiet:
        print("%s%sMoonShadowAI · 生成%s" % (C.BOLD, C.CYAN, C.RESET))
        print("%s  checkpoint : %s%s" % (C.GRAY, _abs(args.ckpt), C.RESET))
        print("%s  epoch/loss: %s / %s" % (C.GRAY, meta.get("epoch"),
                                          ("%.4f" % float(meta["loss"])) if meta.get("loss") is not None else "--"))
        print("%s  架构      : vocab=%s embed=%s hidden=%s layers=%s block=%s" % (
            C.GRAY, meta.get("vocab_size"), meta.get("embed_dim"), meta.get("hidden_dim"),
            meta.get("num_layers"), meta.get("block_size")))
        print("%s  参数量    : %s（%.2f M）" % (C.GRAY, format(count_params(model), ","), count_params(model) / 1e6))
        print("%s  采样      : temp=%s topk=%s topp=%s len=%s n=%s" % (
            C.GRAY, args.temp, args.topk, args.topp, args.len, args.n))
        print("%s  词表接口  : %s%s" % (C.GRAY, codec.describe(), C.RESET))
        print("%s  prompt    : %s%s%s" % (C.GRAY, C.BOLD, (args.prompt or "<随机起头>"), C.RESET))
        print("")

    prompt_ids = [i for i in codec.encode(args.prompt or "") if i >= 4]
    if args.prompt and not prompt_ids:
        print("%s[警告] 提示词「%s」里的字都不在词表内，已退化为随机起头%s" % (C.YELLOW, args.prompt, C.RESET))

    for i in range(max(1, args.n)):
        color = COLORS[i % len(COLORS)]
        seed = None if args.seed is None else args.seed + i
        try:
            text = sample(model, codec, args.prompt, max_len=args.len, temperature=args.temp,
                          top_k=args.topk, top_p=args.topp, device=device, seed=seed)
        except Exception as exc:
            print("%s[错误] 第 %d 条生成失败：%s: %s%s" % (C.RED, i + 1, type(exc).__name__, exc, C.RESET))
            continue
        head = "%s〔%d/%d〕%s" % (C.DIM, i + 1, max(1, args.n), C.RESET) if not args.quiet else ""
        body = "%s%s%s" % (color, text, C.RESET) if not args.quiet else text
        print("%s%s" % (head, body))
    if not args.quiet:
        print("\n%s%s（来自 epoch %s，loss %s）%s" % (
            C.GRAY, "—" * 20, meta.get("epoch"),
            ("%.4f" % float(meta["loss"])) if meta.get("loss") is not None else "?", C.RESET))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n%s[dim]已中断%s" % (C.GRAY, C.RESET))
        sys.stdout.write("\033[0m")
        sys.exit(130)
    finally:
        sys.stdout.write("\033[0m")
        try:
            sys.stdout.flush()
        except Exception:
            pass
