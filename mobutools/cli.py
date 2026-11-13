#!/usr/bin/env python3
"""MobuTools 命令行工具"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobutools import MobuModel
from mobutools.mobu_model import TYPE_NEURAL
from mobutools.neural_io import state_dict_to_mobu
import torch


def cmd_info(args):
    if not args:
        print("用法: mobutool info <model.mobu>")
        return
    filepath = args[0]
    try:
        model = MobuModel.load(filepath)
        info = model.to_dict()
        print(f"文件: {filepath}")
        print(f"魔数: {info['magic']}")
        print(f"版本: v{info['version']}")
        print(f"类型: {info['model_type']}")
        print(f"词表: {info['vocab_size']} 词")
        print(f"Bigram: {info['bigram_entries']} 条")
        print(f"神经网络权重块: {info['neural_blocks']}")
        if info['metadata']:
            print("元数据:")
            for k, v in info['metadata'].items():
                print(f"  {k}: {v}")
        errors = model.validate()
        if errors:
            print("警告:")
            for e in errors:
                print(f"  ⚠ {e}")
        else:
            print("校验: ✅ 通过")
    except Exception as e:
        print(f"错误: {e}")


def cmd_dump(args):
    if not args:
        print("用法: mobutool dump <model.mobu> [output.json]")
        return
    filepath = args[0]
    outpath = args[1] if len(args) > 1 else None
    try:
        model = MobuModel.load(filepath)
        if outpath:
            model.to_json(outpath)
            print(f"已导出: {outpath}")
        else:
            print(model.to_json())
    except Exception as e:
        print(f"错误: {e}")


def cmd_convert(args):
    if len(args) < 2:
        print("用法: mobutool convert <input.mobu> <output.mobu>")
        return
    inpath, outpath = args[0], args[1]
    try:
        model = MobuModel.load(inpath)
        model.save(outpath)
        print(f"已转换: {inpath} -> {outpath}")
        info = model.to_dict()
        print(f"版本: v{info['version']}, 词表: {info['vocab_size']} 词, Bigram: {info['bigram_entries']} 条")
    except Exception as e:
        print(f"错误: {e}")


def cmd_validate(args):
    if not args:
        print("用法: mobutool validate <model.mobu>")
        return
    filepath = args[0]
    try:
        model = MobuModel.load(filepath)
        errors = model.validate()
        if errors:
            print("发现以下问题:")
            for e in errors:
                print(f"  ⚠ {e}")
        else:
            print("✅ 模型文件完整无误")
    except Exception as e:
        print(f"错误: {e}")


def cmd_export(args):
    """把 .pt 检查点转成 .mobu"""
    if len(args) < 2:
        print("用法: mobutool export <model.pt> <out.mobu> [--quantize]")
        return
    in_pt, out_mobu = args[0], args[1]
    quantize = "--quantize" in args

    ckpt = torch.load(in_pt, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    mobu = MobuModel()
    mobu.model_type = TYPE_NEURAL
    mobu.set_meta("model_type", "lstm_language_model")
    mobu.set_meta("framework", "pytorch")
    mobu.set_meta("vocab_size", ckpt.get("vocab_size", 0) if isinstance(ckpt, dict) else 0)
    mobu.set_meta("embed_dim", ckpt.get("embed_dim", 0) if isinstance(ckpt, dict) else 0)
    mobu.set_meta("hidden_dim", ckpt.get("hidden_dim", 0) if isinstance(ckpt, dict) else 0)
    mobu.set_meta("num_layers", ckpt.get("num_layers", 1) if isinstance(ckpt, dict) else 1)
    mobu.set_meta("quantized", 1 if quantize else 0)
    mobu.set_meta("trained_by", "GitHub Actions")

    # 词表也打包进去
    vocab_path = os.path.join(os.path.dirname(out_mobu), "vocab.mobu")
    if os.path.exists(vocab_path):
        vm = MobuModel.load(vocab_path)
        mobu.vocab = vm.vocab
        mobu.word_to_id = vm.word_to_id

    # 把 state_dict 包装成 model 壳
    class WrapModel:
        def __init__(self, d):
            self._d = d
        def state_dict(self):
            return self._d

    wrap = WrapModel(sd)
    state_dict_to_mobu(wrap, mobu, quantize=quantize)
    mobu.save(out_mobu)

    size_mb = os.path.getsize(out_mobu) / 1024 / 1024
    print(f"✅ 导出完成: {out_mobu} ({size_mb:.2f} MB), {len(mobu.neural_data)} 块权重")


def main():
    if len(sys.argv) < 2:
        print("MobuTools v0.1.0 — MoonShadowAI 模型文件工具")
        print()
        print("用法:")
        print("  mobutool info <file.mobu>       查看模型信息")
        print("  mobutool dump <file.mobu> [json] 导出为 JSON")
        print("  mobutool convert <in> <out>      旧版转新版")
        print("  mobutool validate <file.mobu>    校验完整性")
        print("  mobutool export <in.pt> <out>    导出 .pt 到 .mobu")
        return

    cmd = sys.argv[1]
    args = sys.argv[2:]

    commands = {
        "info": cmd_info,
        "dump": cmd_dump,
        "convert": cmd_convert,
        "validate": cmd_validate,
        "export": cmd_export,
    }

    if cmd in commands:
        commands[cmd](args)
    else:
        print(f"未知命令: {cmd}")
        print("可用命令: info, dump, convert, validate, export")


if __name__ == "__main__":
    main()
