# -*- coding: utf-8 -*-
"""MobuModel：自定义二进制容器（词表 + 序列/权重）。"""
import struct

MAGIC = b"MOBU"
VERSION = 2
SEG_META = 0x01
SEG_VOCAB = 0x02
SEG_WEIGHT = 0x03
DTYPE_CODES = {1: "utf-8", 2: "float32", 3: "float64", 4: "int64", 5: "int32", 6: "uint8"}


class MobuModel:
    def __init__(self):
        self.meta = {}
        self.vocab = []            # index -> token
        self.neural_data = []      # {"name","shape","dtype","data"}

    # ---------- 基本操作 ----------
    def set_meta(self, k, v):
        self.meta[str(k)] = str(v)

    def get_meta(self, k):
        return self.meta.get(str(k))

    def add_word(self, w):
        w = str(w)
        if w not in self.vocab:
            self.vocab.append(w)

    def word_to_id(self, w):
        w = str(w)
        try:
            return self.vocab.index(w)
        except ValueError:
            return 0

    def id_to_word(self, i):
        i = int(i)
        if 0 <= i < len(self.vocab):
            return self.vocab[i]
        return self.vocab[0] if self.vocab else "<UNK>"

    def vocab_size(self):
        return len(self.vocab)

    def set_vocab(self, lst):
        self.vocab = [str(t) for t in lst]

    def get_vocab(self):
        return list(self.vocab)

    def add_weight(self, name, tensor_bytes, dtype_code, shape):
        self.neural_data = [b for b in self.neural_data if b["name"] != name]
        self.neural_data.append({
            "name": str(name),
            "shape": [int(d) for d in shape],
            "dtype": int(dtype_code),
            "data": bytes(tensor_bytes),
        })

    def get_weight(self, name):
        for b in self.neural_data:
            if b["name"] == name:
                return b
        return None

    # ---------- 序列化 ----------
    def save(self, path):
        segs = []
        # META
        p = b""
        for k, v in self.meta.items():
            kb = k.encode("utf-8"); vb = v.encode("utf-8")
            p += struct.pack("<I", len(kb)) + kb + struct.pack("<I", len(vb)) + vb
        segs.append(struct.pack("<BQ", SEG_META, len(p)) + p)
        # VOCAB
        p = struct.pack("<I", len(self.vocab))
        for t in self.vocab:
            tb = t.encode("utf-8")
            p += struct.pack("<H", len(tb)) + tb
        segs.append(struct.pack("<BQ", SEG_VOCAB, len(p)) + p)
        # WEIGHT
        for b in self.neural_data:
            nb = b["name"].encode("ascii")
            p = struct.pack("<H", len(nb)) + nb + struct.pack("<BB", b["dtype"], len(b["shape"]))
            for d in b["shape"]:
                p += struct.pack("<i", d)
            p += b["data"]
            segs.append(struct.pack("<BQ", SEG_WEIGHT, len(p)) + p)
        blob = struct.pack("<4sII", MAGIC, VERSION, len(segs)) + b"".join(segs)
        with open(path, "wb") as f:           # 存在即覆盖
            f.write(blob)
        return len(blob)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            blob = f.read()
        m = cls()
        off = 0
        magic, ver, seg_count = struct.unpack_from("<4sII", blob, off)
        off += 12
        if magic != MAGIC:
            raise ValueError("不是 MOBU 文件: %s" % path)
        if ver != VERSION:
            raise ValueError("MOBU 版本不符: %d (期望 %d)" % (ver, VERSION))
        for _ in range(seg_count):
            seg_type, seg_len = struct.unpack_from("<BQ", blob, off)
            off += 9
            pay = blob[off:off + seg_len]
            off += seg_len
            if seg_type == SEG_META:
                o = 0
                while o < len(pay):
                    kl, = struct.unpack_from("<I", pay, o); o += 4
                    k = pay[o:o + kl].decode("utf-8"); o += kl
                    vl, = struct.unpack_from("<I", pay, o); o += 4
                    v = pay[o:o + vl].decode("utf-8"); o += vl
                    m.meta[k] = v
            elif seg_type == SEG_VOCAB:
                o = 0
                cnt, = struct.unpack_from("<I", pay, o); o += 4
                m.vocab = []
                for _i in range(cnt):
                    tl, = struct.unpack_from("<H", pay, o); o += 2
                    m.vocab.append(pay[o:o + tl].decode("utf-8")); o += tl
            elif seg_type == SEG_WEIGHT:
                o = 0
                nl, = struct.unpack_from("<H", pay, o); o += 2
                name = pay[o:o + nl].decode("ascii"); o += nl
                dtype, ndim = struct.unpack_from("<BB", pay, o); o += 2
                shape = []
                for _d in range(ndim):
                    shape.append(struct.unpack_from("<i", pay, o)[0]); o += 4
                m.add_weight(name, pay[o:], dtype, shape)
            else:
                raise ValueError("未知段类型 0x%02X" % seg_type)
        return m

    # ---------- 展示 ----------
    def summary(self):
        lines = ["MobuModel  %s" % "(内存对象)", "  魔数      : MOBU / v%d" % VERSION,
                 "  元数据    : %d 条" % len(self.meta), "  词表      : %d 词" % len(self.vocab)]
        if self.neural_data:
            lines.append("  权重块    : %d" % len(self.neural_data))
            for b in self.neural_data:
                n = 1
                for d in b["shape"]:
                    n *= d
                lines.append("    - %-28s %-12s %-10s elems=%d bytes=%d" %
                             (b["name"], "x".join(str(d) for d in b["shape"]),
                              DTYPE_CODES.get(b["dtype"], "code%d" % b["dtype"]), n, len(b["data"])))
        else:
            lines.append("  权重块    : 0")
        for k, v in self.meta.items():
            lines.append("    meta %s = %s" % (k, v))
        return "\n".join(lines)

    def to_dict(self):
        weights = []
        for b in self.neural_data:
            weights.append({"name": b["name"], "shape": b["shape"], "dtype_code": b["dtype"]})
        return {
            "vocab_size": len(self.vocab),
            "weights": weights,
            "vocab_sample": list(self.vocab[:10]),
            "metadata": dict(self.meta),
        }
