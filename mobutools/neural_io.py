"""神经网络权重段的读写（Mobu v2 扩展）"""

import struct
import torch
import numpy as np

DTYPE_MAP = {
    "float32": 0x01, "float16": 0x02, "int8": 0x03,
    "int64": 0x04, "bfloat16": 0x05,
}
DTYPE_TORCH = {
    0x01: torch.float32, 0x02: torch.float16, 0x03: torch.int8,
    0x04: torch.int64, 0x05: torch.bfloat16,
}
# 新增：Torch dtype → NumPy dtype 字符串
TORCH_TO_NUMPY = {
    torch.float32: "float32", torch.float16: "float16",
    torch.int8: "int8", torch.int64: "int64",
    torch.bfloat16: "bfloat16",
}
DTYPE_NBYTES = {0x01: 4, 0x02: 2, 0x03: 1, 0x04: 8, 0x05: 2}


def state_dict_to_mobu(model, mobu, prefix_filter=None, quantize=False):
    """把 PyTorch 模型的 state_dict 塞进 MobuModel"""
    sd = model.state_dict()
    for name, tensor in sd.items():
        if prefix_filter and not name.startswith(prefix_filter):
            continue
        t = tensor.detach().cpu().contiguous()

        if quantize and t.dtype in (torch.float32, torch.float16) and t.dim() >= 2:
            absmax = t.abs().max().item()
            scale = absmax / 127.0 if absmax > 0 else 1.0
            q = torch.clamp(torch.round(t / scale), -128, 127).to(torch.int8)
            mobu.set_meta(f"scale_{name}", scale)
            dtype_code = 0x03
            raw = q.numpy().tobytes()
        else:
            dtype_code = DTYPE_MAP.get(TORCH_TO_NUMPY[t.dtype], 0x01)
            raw = t.numpy().tobytes()

        mobu.add_weight(name=name, tensor_bytes=raw,
                        dtype_code=dtype_code, shape=list(t.shape))


def mobu_to_state_dict(mobu):
    """从 MobuModel 还原 state_dict"""
    sd = {}
    for blk in mobu.neural_data:
        torch_dtype = DTYPE_TORCH[blk["dtype"]]
        # 用 torch_dtype 转成 numpy dtype 字符串
        np_dtype_str = TORCH_TO_NUMPY[torch_dtype]
        arr = np.frombuffer(bytearray(blk["data"]), dtype=np.dtype(np_dtype_str))
        t = torch.from_numpy(arr.copy()).reshape(blk["shape"])
        if blk["dtype"] == 0x03:
            scale = mobu.get_meta(f"scale_{blk['name']}")
            if scale is not None:
                t = t.to(torch.float32) * scale
        sd[blk["name"]] = t
    return sd
