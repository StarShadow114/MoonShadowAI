"""
MobuModel — 新版 MOBU 模型文件读写
魔数: MOBU (4字节)
版本: 0x0002 (2字节)
模型类型: 0x01=Bigram, 0x02=NeuralNet (1字节)
元数据: KV 对序列
校验区: CRC32 (4字节)
词库: 旧版兼容格式
模型数据: 按类型区分
"""

import struct
import json
import zlib
import os


MAGIC = b"MOBU"
VERSION = 0x0002

TYPE_BIGRAM = 0x01
TYPE_NEURAL = 0x02

KV_TYPE_STR = 0x01
KV_TYPE_INT = 0x02
KV_TYPE_FLOAT = 0x03

DTYPE_FLOAT32 = 0x01
DTYPE_FLOAT16 = 0x02
DTYPE_INT8 = 0x03


class MobuError(Exception):
    pass


class MobuModel:
    def __init__(self):
        self.version = VERSION
        self.model_type = TYPE_BIGRAM
        self.metadata = {}  # {key: (type_str, value)}
        self.vocab = {}     # {id: word}
        self.word_to_id = {}  # {word: id}
        self.bigram = {}    # {first: {second: {"weight": int}}}
        self.neural_data = []  # 神经网络权重块列表

    # ---- 元数据操作 ----
    def set_meta(self, key, value):
        if isinstance(value, str):
            self.metadata[key] = ("str", value)
        elif isinstance(value, int):
            self.metadata[key] = ("int", value)
        elif isinstance(value, float):
            self.metadata[key] = ("float", value)
        else:
            raise MobuError(f"不支持的元数据类型: {type(value)}")

    def get_meta(self, key):
        if key not in self.metadata:
            return None
        return self.metadata[key][1]

    # ---- 词表操作 ----
    def add_word(self, word):
        if word in self.word_to_id:
            return self.word_to_id[word]
        idx = len(self.vocab)
        self.vocab[idx] = word
        self.word_to_id[word] = idx
        return idx

    def get_word(self, idx):
        return self.vocab.get(idx, "")

    def get_id(self, word):
        return self.word_to_id.get(word, -1)

    # ---- Bigram 操作 ----
    def add_bigram_entry(self, first_word, second_word, weight=1):
        if first_word not in self.bigram:
            self.bigram[first_word] = {}
        self.bigram[first_word][second_word] = {"weight": weight}

    def get_bigram_weight(self, first_word, second_word):
        if first_word in self.bigram and second_word in self.bigram[first_word]:
            return self.bigram[first_word][second_word]["weight"]
        return 0

    # ---- 神经网络权重操作 ----
    def add_weight(self, name, tensor_bytes, dtype_code, shape):
        """添加一块权重"""
        self.neural_data.append({
            "name": name,
            "data": tensor_bytes,
            "dtype": dtype_code,
            "shape": list(shape),
        })

    # ---- 序列化 ----
    def _serialize_metadata(self):
        buf = b""
        for key, (typ, val) in self.metadata.items():
            key_bytes = key.encode("utf-8")
            if typ == "str":
                val_bytes = val.encode("utf-8")
                buf += struct.pack("<BH", KV_TYPE_STR, len(key_bytes))
                buf += key_bytes
                buf += struct.pack("<I", len(val_bytes))
                buf += val_bytes
            elif typ == "int":
                buf += struct.pack("<BH", KV_TYPE_INT, len(key_bytes))
                buf += key_bytes
                buf += struct.pack("<Ii", 4, val)
            elif typ == "float":
                buf += struct.pack("<BH", KV_TYPE_FLOAT, len(key_bytes))
                buf += key_bytes
                buf += struct.pack("<If", 4, val)
        return buf

    def _serialize_vocab(self):
        buf = b""
        for idx in sorted(self.vocab.keys()):
            word = self.vocab[idx]
            word_bytes = word.encode("utf-8")
            buf += struct.pack("<IH", idx, len(word_bytes))
            buf += word_bytes
        return buf

    def _serialize_bigram(self):
        buf = b""
        for first, followers in self.bigram.items():
            first_id = self.word_to_id.get(first, -1)
            if first_id < 0:
                continue
            for second, data in followers.items():
                second_id = self.word_to_id.get(second, -1)
                if second_id < 0:
                    continue
                weight = data["weight"]
                buf += struct.pack("<IIBB", first_id, second_id, weight, 0)
        return buf

    def _serialize_neural(self):
        buf = b""
        for blk in self.neural_data:
            name_b = blk["name"].encode("utf-8")
            buf += struct.pack("<H", len(name_b))
            buf += name_b
            buf += struct.pack("<BH", blk["dtype"], len(blk["shape"]))
            for d in blk["shape"]:
                buf += struct.pack("<I", d)
            buf += struct.pack("<I", len(blk["data"]))
            buf += blk["data"]
        return buf

    def save(self, filepath):
        metadata_bytes = self._serialize_metadata()
        vocab_bytes = self._serialize_vocab()
        bigram_bytes = self._serialize_bigram()
        neural_bytes = self._serialize_neural()

        # 组装主体数据（用于计算 CRC）
        body = b""
        body += struct.pack("<HB", self.version, self.model_type)
        body += struct.pack("<I", len(metadata_bytes))
        body += metadata_bytes
        body += struct.pack("<I", len(vocab_bytes))
        body += vocab_bytes
        body += struct.pack("<I", len(bigram_bytes))
        body += bigram_bytes
        body += struct.pack("<I", len(neural_bytes))
        body += neural_bytes

        crc = zlib.crc32(body) & 0xFFFFFFFF

        with open(filepath, "wb") as f:
            f.write(MAGIC)
            f.write(body)
            f.write(struct.pack("<I", crc))

    # ---- 反序列化 ----
    @staticmethod
    def load(filepath):
        if not os.path.exists(filepath):
            raise MobuError(f"文件不存在: {filepath}")

        with open(filepath, "rb") as f:
            data = f.read()

        # 检查是否为旧版（无魔数）
        if len(data) < 4 or data[:4] != MAGIC:
            return MobuModel._load_legacy(filepath, data)

        model = MobuModel()
        pos = 4

        # 版本号和模型类型
        model.version, model.model_type = struct.unpack_from("<HB", data, pos)
        pos += 3

        # 元数据
        meta_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        meta_end = pos + meta_len
        while pos < meta_end:
            kv_type, key_len = struct.unpack_from("<BH", data, pos)
            pos += 3
            key = data[pos:pos+key_len].decode("utf-8")
            pos += key_len
            val_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            if kv_type == KV_TYPE_STR:
                val = data[pos:pos+val_len].decode("utf-8")
            elif kv_type == KV_TYPE_INT:
                val = struct.unpack_from("<i", data, pos)[0]
            elif kv_type == KV_TYPE_FLOAT:
                val = struct.unpack_from("<f", data, pos)[0]
            else:
                val = None
            pos += val_len
            if val is not None:
                type_map = {KV_TYPE_STR: "str", KV_TYPE_INT: "int", KV_TYPE_FLOAT: "float"}
                model.metadata[key] = (type_map.get(kv_type, "str"), val)

        # 词库
        vocab_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        vocab_end = pos + vocab_len
        while pos < vocab_end:
            idx, word_len = struct.unpack_from("<IH", data, pos)
            pos += 6
            word = data[pos:pos+word_len].decode("utf-8")
            pos += word_len
            model.vocab[idx] = word
            model.word_to_id[word] = idx

        # Bigram 数据
        bigram_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        bigram_end = pos + bigram_len
        while pos < bigram_end:
            first_id, second_id, weight, _ = struct.unpack_from("<IIBB", data, pos)
            pos += 10
            first = model.vocab.get(first_id, "")
            second = model.vocab.get(second_id, "")
            if first and second:
                if first not in model.bigram:
                    model.bigram[first] = {}
                model.bigram[first][second] = {"weight": weight}

        # 神经网络数据
        neural_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        neural_end = pos + neural_len
        while pos < neural_end:
            name_len = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            name = data[pos:pos+name_len].decode("utf-8")
            pos += name_len
            dtype_code, ndim = struct.unpack_from("<BH", data, pos)
            pos += 3
            shape = []
            for _ in range(ndim):
                d = struct.unpack_from("<I", data, pos)[0]
                pos += 4
                shape.append(d)
            data_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            raw = data[pos:pos+data_len]
            pos += data_len
            model.neural_data.append({
                "name": name,
                "data": raw,
                "dtype": dtype_code,
                "shape": shape,
            })

        # 校验 CRC
        body_len = pos - 4
        expected_crc = struct.unpack_from("<I", data, pos)[0]
        actual_crc = zlib.crc32(data[4:pos]) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise MobuError(f"CRC 校验失败: 期望 {expected_crc:#x}, 实际 {actual_crc:#x}")

        return model

    @staticmethod
    def _load_legacy(filepath, data):
        """兼容旧版 mobu 文件（无魔数、无版本号）"""
        model = MobuModel()
        model.version = 0x0001
        model.model_type = TYPE_BIGRAM

        pos = 0
        while pos < len(data):
            if pos + 6 > len(data):
                break
            idx, word_len = struct.unpack_from("<IH", data, pos)
            pos += 6
            if pos + word_len > len(data):
                break
            word = data[pos:pos+word_len].decode("utf-8")
            pos += word_len
            model.vocab[idx] = word
            model.word_to_id[word] = idx

        # 剩余部分为 bigram 数据
        while pos + 10 <= len(data):
            first_id, second_id, weight, _ = struct.unpack_from("<IIBB", data, pos)
            pos += 10
            first = model.vocab.get(first_id, "")
            second = model.vocab.get(second_id, "")
            if first and second:
                if first not in model.bigram:
                    model.bigram[first] = {}
                model.bigram[first][second] = {"weight": weight}

        model.set_meta("source", "legacy_format")
        return model

    # ---- 导出 ----
    def to_dict(self):
        result = {
            "magic": "MOBU",
            "version": self.version,
            "model_type": "bigram" if self.model_type == TYPE_BIGRAM else "neural",
            "metadata": {},
            "vocab_size": len(self.vocab),
            "bigram_entries": sum(len(f) for f in self.bigram.values()),
            "neural_blocks": len(self.neural_data),
        }
        for key, (typ, val) in self.metadata.items():
            result["metadata"][key] = val
        return result

    def to_json(self, filepath=None):
        data = self.to_dict()
        data["vocab"] = {str(k): v for k, v in sorted(self.vocab.items())}
        data["bigram"] = {}
        for first, followers in self.bigram.items():
            data["bigram"][first] = {}
            for second, info in followers.items():
                data["bigram"][first][second] = info["weight"]
        data["neural"] = []
        for blk in self.neural_data:
            data["neural"].append({
                "name": blk["name"],
                "dtype": blk["dtype"],
                "shape": blk["shape"],
                "size_bytes": len(blk["data"]),
            })

        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    # ---- 校验 ----
    def validate(self):
        errors = []
        if self.version == 0x0001:
            errors.append("旧版格式（v1），建议转换为 v2")
        if self.model_type not in (TYPE_BIGRAM, TYPE_NEURAL):
            errors.append(f"未知模型类型: {self.model_type}")
        if not self.vocab:
            errors.append("词表为空")
        if self.model_type == TYPE_BIGRAM and not self.bigram:
            errors.append("Bigram 模型但无 bigram 数据")
        ids = sorted(self.vocab.keys())
        if ids and ids[-1] != len(ids) - 1:
            errors.append(f"词表 ID 不连续: 最大 {ids[-1]}, 应有 {len(ids)-1}")
        return errors
