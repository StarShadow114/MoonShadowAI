# -*- coding: utf-8 -*-
"""mobutool —— MoonShadowAI .mobu 容器工具（info / dump / pt2mobu）

子命令：
    info      file.mobu                     查看容器概况并校验
    dump      file.mobu [out.json]          导出 to_dict 为 JSON
    pt2mobu   checkpoint.pt [out.mobu]      PyTorch checkpoint → 自包含 .mobu

用法示例：
    python -m mobutools.cli info neural/data/vocab.mobu
    python -m mobutools.cli dump neural/data/vocab.mobu vocab.json
    python -m mobutools.cli pt2mobu neural/checkpoints/best_model.pt
    python -m mobutools.cli pt2mobu ckpt.pt --quantize --meta note=CPU测试 --force
    python -m mobutools.cli pt2mobu --help
"""
import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from mobutools import (MobuModel, MAGIC, VERSION, DTYPE_CODES,
                       state_dict_to_mobu_existing, mobu_to_state_dict)

ARCH_KEYS = ("vocab_size", "embed_dim", "hidden_dim", "num_layers", "block_size")


def _abs(p):
    p = os.path.expanduser(str(p or ""))
    return p if (not p or os.path.isabs(p)) else os.path.join(PROJECT_ROOT, p)


def _numel(shape):
    n = 1
    for d in shape or ():
        n *= int(d)
    return n


def _dtype_name(code):
    return DTYPE_CODES.get(code, "code%s" % code)


# ---------------------------------------------------------------- info
def cmd_info(args):
    path = _abs(args.file)
    if not os.path.exists(path):
        print("\033[31m[错误] 找不到文件：%s\033[0m" % path)
        return 1
    with open(path, "rb") as f:
        head = f.read(12)
    errors = []
    mobu = MobuModel.load(path)
    magic = head[:4]
    ver = int.from_bytes(head[4:8], "little") if len(head) >= 8 else -1
    seg_count = int.from_bytes(head[8:12], "little") if len(head) >= 12 else -1

    print("\033[1m\033[36m▸ %s\033[0m" % path)
    print("  大小      : %s 字节" % format(os.path.getsize(path), ","))
    print("  魔数      : %r %s" % (magic, "✓" if magic == MAGIC else "✗ 期望 %r" % MAGIC))
    print("  版本      : %d %s" % (ver, "✓" if ver == VERSION else "⚠ 本工具版本 %d" % VERSION))
    print("  段数      : %d（声明 %d）" % (1 + 1 + len(mobu.neural_data) if mobu.vocab or mobu.neural_data
                                           else len(mobu.neural_data), seg_count))
    print("  词表      : %d 词%s" % (mobu.vocab_size(),
                                     "（样例：%s …）" % "、".join(mobu.vocab[:8]) if mobu.vocab else "（无）"))
    if mobu.neural_data:
        total = 0
        print("  权重块    : %d" % len(mobu.neural_data))
        for b in mobu.neural_data:
            n = _numel(b["shape"])
            total += n
            scale = mobu.get_meta("scale_%s" % b["name"])
            q = "（int8 量化，scale=%s）" % scale if scale is not None and b["dtype"] == 0x03 else ""
            print("    - %-24s %-10s %-22s 元素 %s%s" % (
                b["name"], _dtype_name(b["dtype"]), list(b["shape"]), format(n, ","), q))
            expect = n * {0x01: 4, 0x02: 2, 0x03: 1, 0x04: 8, 0x05: 2, 6: 1}.get(b["dtype"], 0)
            if expect != len(b["data"]):
                errors.append("权重 %s 字节数 %d ≠ 期望 %d" % (b["name"], len(b["data"]), expect))
        print("  参数总数  : %s" % format(total, ","))
    if mobu.meta:
        print("  元数据    : %d 条" % len(mobu.meta))
        for k, v in mobu.meta.items():
            print("    %s = %s" % (k, v))
    try:
        sd = mobu_to_state_dict(mobu)
        for name, t in sd.items():
            if not torch.isfinite(t.float()).all():
                errors.append("权重 %s 含 NaN/Inf" % name)
        print("  校验      : %s（%d 个张量可还原、有限值）" % (
            "\033[32mOK ✓\033[0m" if not errors else "\033[31m发现 %d 个问题 ✗\033[0m" % len(errors), len(sd)))
    except Exception as exc:
        print("  校验      : \033[31m还原失败：%s: %s\033[0m" % (type(exc).__name__, exc))
        return 1
    for e in errors:
        print("  \033[33m⚠ %s\033[0m" % e)
    return 1 if errors else 0


# ---------------------------------------------------------------- dump
def cmd_dump(args):
    path = _abs(args.file)
    if not os.path.exists(path):
        print("\033[31m[错误] 找不到文件：%s\033[0m" % path)
        return 1
    mobu = MobuModel.load(path)
    data = mobu.to_dict()
    out = _abs(args.output) if args.output else (os.path.splitext(path)[0] + ".json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print("\033[32m✓\033[0m 已导出 %s（%s 词，%d 个权重块）" % (
        out, data.get("vocab_size"), len(data.get("weights", []))))
    return 0


# ---------------------------------------------------------------- pt2mobu
def _probe_vocab(arg, ckpt_dir):
    """词表探测：显式参数 → checkpoint 同目录 → neural/data/ → 包目录。"""
    cands = []
    if arg:
        cands.append(arg)
    cands += [
        os.path.join(ckpt_dir, "vocab.mobu"),
        os.path.join("neural", "data", "vocab.mobu"),
        os.path.join(PROJECT_ROOT, "neural", "data", "vocab.mobu"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.mobu"),
    ]
    seen = set()
    for c in cands:
        p = _abs(c) if not os.path.isabs(c) else c
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            return p
    return None


def cmd_pt2mobu(args):
    src = _abs(args.checkpoint)
    if not os.path.exists(src):
        print("\033[31m[错误] 找不到 checkpoint：%s\033[0m" % src)
        return 1
    out = _abs(args.output or args.out_opt) if (args.output or args.out_opt) else os.path.splitext(src)[0] + ".mobu"
    if os.path.exists(out) and not args.force:
        print("\033[31m[错误] %s 已存在，加 --force 覆盖\033[0m" % out)
        return 1

    obj = torch.load(src, map_location="cpu")
    if not isinstance(obj, dict):
        print("\033[31m[错误] checkpoint 内容不是 dict：%s\033[0m" % type(obj).__name__)
        return 1
    if "model_state_dict" in obj:
        sd_raw, meta = obj["model_state_dict"], obj
    else:
        sd_raw, meta = obj, {}
        print("\033[33m⚠ 未找到 model_state_dict，按“整个文件即 state_dict”处理\033[0m")
    sd = {k: v.detach().cpu() for k, v in sd_raw.items() if hasattr(v, "detach")}
    if not sd:
        print("\033[31m[错误] 没从 checkpoint 里取出任何张量\033[0m")
        return 1

    cfg = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    arch = {}
    for k in ARCH_KEYS:
        v = meta.get(k, cfg.get("model.%s" % k if k != "vocab_size" else "model.vocab_size"))
        if v is None and k == "vocab_size":
            for key in ("weight", "embed_weight", "embedding.weight", "encoder.weight"):
                if key in sd:
                    v = sd[key].shape[0]
                    break
        if v is not None:
            arch[k] = int(v)
    vocab_size = arch.get("vocab_size")
    if vocab_size is None:
        for name, t in sd.items():
            if name.endswith("weight") and t.dim() == 2:
                vocab_size = int(t.shape[0])
                print("\033[33m⚠ checkpoint 无 vocab_size，从 %s 推断为 %d\033[0m" % (name, vocab_size))
                break
    if vocab_size is None:
        print("\033[31m[错误] 无法确定 vocab_size，请用 --meta vocab_size=N 指定\033[0m")
        return 1

    mobu = MobuModel()
    mobu.set_meta("source", os.path.basename(src))
    mobu.set_meta("source_epoch", meta.get("epoch", "?"))
    mobu.set_meta("source_loss", "%.6f" % float(meta.get("loss", 0.0)) if meta.get("loss") is not None else "?")
    mobu.set_meta("converted_by", "mobutool pt2mobu (MOBU v%d)" % VERSION)
    mobu.set_meta("quantized", "int8" if args.quantize else "none")
    for k, v in arch.items():
        mobu.set_meta(k, v)
    if cfg:
        for k, v in cfg.items():
            mobu.set_meta("cfg.%s" % k, v)
    for kv in (args.meta or ()):
        if "=" not in kv:
            print("\033[33m⚠ 忽略非法 --meta（缺 =）：%s\033[0m" % kv)
            continue
        k, v = kv.split("=", 1)
        mobu.set_meta(k.strip(), v.strip())
        print("  meta %s = %s" % (k.strip(), v.strip()))
    # ---- 词表 ----
    vocab_path = None
    if not args.no_vocab:
        vocab_path = _probe_vocab(args.vocab, os.path.dirname(src))
        if vocab_path:
            try:
                vm = MobuModel.load(vocab_path)
                vlist = vm.get_vocab() if hasattr(vm, "get_vocab") else []
                if vlist:
                    mobu.set_vocab(list(vlist))
                    for k, v in vm.meta.items():
                        if k not in mobu.meta:
                            mobu.set_meta("vocab_%s" % k, v)
                    print("\033[32m✓\033[0m 词表：set_vocab 整表写入（%d 词，来自 %s）" % (len(vlist), vocab_path))
                else:
                    vocab_path = None
            except Exception as exc:
                print("\033[33m⚠ 词表读取失败（%s），退化为按 id 占位\033[0m" % exc)
                vocab_path = None
        if not vocab_path:
            for i in range(vocab_size):
                mobu.add_word("<id%d>" % i)
            print("\033[33m⚠ 未找到全量词表，已按 vocab_size=%d 写入占位 token，"
                  "用 --vocab 指定 vocab.mobu 可获得真实文本\033[0m" % vocab_size)

    # ---- 权重 ----
    state_dict_to_mobu_existing(sd, mobu, prefix_filter=args.prefix or None, quantize=args.quantize)
    if args.prefix:
        print("  仅导出前缀 %r 的张量：%d 个" % (args.prefix, len(mobu.neural_data)))
    mobu.save(out)

    # ---- 回读校验 ----
    chk = MobuModel.load(out)
    n_w = len(chk.neural_data)
    n_v = chk.vocab_size()
    problems = []
    if n_w != len(mobu.neural_data):
        problems.append("权重块数 %d ≠ 写入 %d" % (n_w, len(mobu.neural_data)))
    if not args.no_vocab and n_v != vocab_size:
        problems.append("词表数 %d ≠ vocab_size %d" % (n_v, vocab_size))
    try:
        sd2 = mobu_to_state_dict(chk)
        for name, t in sd2.items():
            if not torch.isfinite(t.float()).all():
                problems.append("权重 %s 含 NaN/Inf" % name)
        rt = "回读 %d 个张量可还原" % len(sd2)
    except Exception as exc:
        problems.append("回读还原失败：%s" % exc)
        rt = "回读还原失败"
    print("\n\033[1m\033[36m▸ %s\033[0m" % out)
    print("  大小      : %s 字节（%s）" % (format(os.path.getsize(out), ","),
                                         "int8 量化" if args.quantize else "全精度"))
    print("  权重块    : %d（%s）" % (n_w, rt))
    print("  词表      : %d" % n_v)
    print("  元数据    : %d 条" % len(chk.meta))
    if problems:
        print("  校验      : \033[31m✗ 警告 %s\033[0m" % "；".join(problems))
        return 1
    print("  校验      : \033[32mOK ✓\033[0m 权重块数与词表数一致")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m mobutools.cli",
        description="mobutool：MOBU 容器检查/导出与 PyTorch checkpoint 转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__ or "")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("info", help="查看 .mobu 概况并校验")
    p1.add_argument("file")
    p1.set_defaults(fn=cmd_info)

    p2 = sub.add_parser("dump", help="导出 JSON")
    p2.add_argument("file")
    p2.add_argument("output", nargs="?", default=None)
    p2.set_defaults(fn=cmd_dump)

    p3 = sub.add_parser("pt2mobu", help="PyTorch checkpoint → .mobu")
    p3.add_argument("checkpoint")
    p3.add_argument("output", nargs="?", default=None, help="输出 .mobu 路径")
    p3.add_argument("-o", dest="out_opt", default=None, help="同位置参数，输出 .mobu 路径")
    p3.add_argument("--vocab", default=None, help="词表 .mobu（默认自动探测）")
    p3.add_argument("--no-vocab", action="store_true", help="只导权重")
    p3.add_argument("--quantize", action="store_true", help="int8 量化（absmax/127，scale 入元数据）")
    p3.add_argument("--meta", action="append", default=[], help="追加元数据 k=v，可多次")
    p3.add_argument("--prefix", default=None, help="只导出该前缀的张量")
    p3.add_argument("--force", action="store_true", help="覆盖已存在输出")
    p3.set_defaults(fn=cmd_pt2mobu)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except Exception as exc:
        print("\033[31m[错误] %s: %s\033[0m" % (type(exc).__name__, exc))
        import traceback
        for line in traceback.format_exc().strip().splitlines()[-8:]:
            print("\033[90m%s\033[0m" % line)
        return 1
    finally:
        sys.stdout.write("\033[0m")
        try:
            sys.stdout.flush()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
