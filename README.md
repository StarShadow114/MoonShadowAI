# MoonShadowAI

从零手写的 LSTM 中文语言模型（纯 CPU，仅 PyTorch + Textual + Rich，不用 HuggingFace）。

## 快速开始
```bash
pip install -r requirements.txt
python scripts/prepare_data.py              # 语料 → vocab.mobu / sequences.mobu
python scripts/train.py -E train.epochs=30  # Textual TUI 训练（非 TTY 自动降级为纯文本）
python scripts/generate.py -p 深度学习 -n 5 # 采样生成
python -m mobutools.cli pt2mobu neural/checkpoints/best_model.pt --quantize --force
python -m mobutools.cli info  neural/checkpoints/best_model.mobu
```

## 结构
- `mobutools/`：MobuModel 二进制容器（MOBU，META/VOCAB/WEIGHT 段）、torch 权重互导、int8 量化、CLI
- `neural/`：模型、配置、数据集、Autotune、仪表盘纯函数、TUI 共用组件
- `scripts/`：数据准备、训练 TUI、生成

## TUI 命令
`help` `pause` / `resume` `status` `lr <值>` `gen [提示词]` `save [路径]` `clear`；
Ctrl+C 第一次暂停、第二次退出；Ctrl+Q 直接退出。

## 说明
内置示例语料约 800 字，只用于跑通流程；换更大语料（prepare_data.py 传文本路径）后 loss 才会真正下降。
`neural/data/` 里已附带示例语料生成的两个 .mobu；`best_model.pt` 由本地跑一次 train.py 生成，再执行 generate.py 与 pt2mobu。
