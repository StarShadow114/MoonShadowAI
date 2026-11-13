# MoonShadow AI

A tiny language model built from scratch by StarShadowStudio.

用最少的代码，搞懂语言模型到底是怎么工作的。

## 当前进度

- [x] v0.1  n-gram 中文文本生成（终端版）
- [x] v0.2  分词器 + 概率模型持久化
- [x] v0.3  模型存储格式 MoBu（二进制紧凑格式）
- [ ] v1.0  真正能聊天的模型

## 模型文件格式

MoonShadowAI 使用自研的 MoBu 格式存储模型文件，位于 models/ 目录：

- vocab.mobu — 词表文件，二进制格式
- bigram.mobu — Bigram 映射文件，二进制格式

MoBu 格式特点：紧凑、安全、解析快。详情见独立仓库 mobu-format。

## 怎么跑

### 训练模型

cd training
python save_model.py

### 生成文本

cd AISRC
python sentence_generator.py

输入起始词即可生成句子，输入 exit 退出。

### 人工反馈训练

cd training
python train_with_feedback.py

程序会自动选取起始词生成句子，你给它打分：
- 输入 p 加分 — 告诉模型"这类句子多来点"
- 输入 m 减分 — 告诉模型"这类句子少来点"
- 直接回车跳过

## 贡献方式

### 提交 PR

欢迎任何人提交 Pull Request。提交前请确保：

1. 你的改动经过测试，不会破坏现有功能
2. 代码风格尽量和已有代码保持一致
3. 如果你改了训练脚本或生成逻辑，请在 PR 描述里说明改动对模型行为的影响

### 参与训练

运行 python training/train_with_feedback.py 可以对模型进行人工反馈训练。

注意：你的每一次打分都会直接写入模型文件，永久改变模型的生成倾向。轻率的打分会让模型产生垃圾输出，请认真对待每一次判断。

### 扩充语料

如果你想让模型学会新的说话风格或词汇，可以把纯文本文件放进 training/materials/ 目录，然后运行 python training/save_model.py 重新训练。

支持的格式：UTF-8 编码的 .txt 文件，内容不限（小说、新闻、对话都可以）。
