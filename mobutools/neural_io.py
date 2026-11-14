# -*- coding: utf-8 -*-
"""PyTorch state_dict <-> MobuModel 权重段互转，支持 int8 量化。"""
import gc

import numpy as np
import torch

DTYPE_MAP = {"float32": 0x01, "float16": 0x02, "int8": 0x03, "int64": 0x04, "bfloat16": 0x05}
DTYPE_TORCH = {0x01: torch.float32, 0x02: torch.float16, 0x03: torch.int8,
               0x04: torch.int64, 0x05: torch.bfloat16}
TORCH_TO_NUMPY = {torch.float32: "float32", torch.float16: "float16", torch.int8: "int8",
                  torch.int64: "int64", torch.bfloat16: "bfloat16"}
DTYPE_NBYTES = {0x01: 4, 0x02: 2, 0x03: 1, 0x04: 8, 0x05: 2}


def _iter_tensors(sd, prefix_filter=None):
    for name, t in sd.items():
        if prefix_filter is None or name.startswith(prefix_filter):
            yield name, t


def _tensor_bytes(name, t, quantize, mobu):
    t = t.detach().cpu().contiguous()
    if quantize and t.is_floating_point() and t.dim() >= 2:
        absmax = t.abs().max().item()
        scale = absmax / 127.0 if absmax != 0 else 1.0
        q = torch.clamp(torch.round(t / scale), -128, 127).to(torch.int8)
        mobu.set_meta("scale_%s" % name, repr(float(scale)))
        return q.numpy().tobytes(), 0x03
    dt = t.dtype
    if dt in (torch.float16, torch.bfloat16):
        t = t.to(torch.float32)
        dt = torch.float32
    if dt not in TORCH_TO_NUMPY:
        t = t.to(torch.float32)
        dt = torch.float32
    np_name = TORCH_TO_NUMPY[dt]
    return t.numpy().astype(np_name).tobytes(), DTYPE_MAP[np_name]


def state_dict_to_mobu_existing(sd, mobu, prefix_filter=None, quantize=False):
    for name, t in _iter_tensors(sd, prefix_filter):
        raw, code = _tensor_bytes(name, t, quantize, mobu)
        mobu.add_weight(name=name, tensor_bytes=raw, dtype_code=code, shape=list(t.shape))


def state_dict_to_mobu(model, mobu, prefix_filter=None, quantize=False):
    state_dict_to_mobu_existing(model.state_dict(), mobu, prefix_filter=prefix_filter, quantize=quantize)


def mobu_to_state_dict(mobu):
    sd = {}
    for blk in mobu.neural_data:
        code = blk["dtype"]
        dt = DTYPE_TORCH[code]
        if code == 0x03:
            np_dtype = "int8"
        elif code in (0x02, 0x05):        # 存时已统一转 float32
            np_dtype = "float32"
            dt = torch.float32
        else:
            np_dtype = TORCH_TO_NUMPY[dt]
        raw = bytearray(blk["data"])
        expect = 1
        for d in blk["shape"]:
            expect *= d
        expect *= DTYPE_NBYTES[code]
        if len(raw) != expect:
            raise ValueError("权重 %s 字节数不符: %d != %d" % (blk["name"], len(raw), expect))
        arr = np.frombuffer(raw, dtype=np_dtype).copy()
        t = torch.from_numpy(arr).reshape(blk["shape"]).to(dt)
        if code == 0x03:
            s = mobu.get_meta("scale_%s" % blk["name"])
            if s is None:
                raise ValueError("量化权重 %s 缺少 scale_%s 元数据" % (blk["name"], blk["name"]))
            t = t.to(torch.float32) * float(s)
        sd[blk["name"]] = t
    gc.collect()
    return sd
