import os
import jieba
import string

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "materials")
extensions = [".md", ".markdown", ".txt"]

files = []
for filename in os.listdir(folder_path):
    if any(filename.endswith(ext) for ext in extensions):
        files.append(filename)
print(files)


def read_books(folder_path, files):
    all_readed_tokens = []
    for i in files:
        path = folder_path + "/" + i
        with open(path, "r", encoding="utf-8") as rf:
            content = rf.read()
        readed_tokens = jieba.lcut(content)
        all_readed_tokens.extend(readed_tokens)
        print(f"共计{len(readed_tokens)}个词")
    return all_readed_tokens


def build_vocab_and_bigram(all_readed_tokens):
    # 词频统计
    tokens_count = {}
    for readed_tokens in all_readed_tokens:
        if readed_tokens in tokens_count:
            tokens_count[readed_tokens] += 1
        else:
            tokens_count[readed_tokens] = 1

    # 数据清洗
    punctuation = string.punctuation + "。，、；：？！「」『』（）【】《》—…·～"
    clean_count = {}
    for word, count in tokens_count.items():
        if len(word) < 2:
            continue
        if all(char in punctuation for char in word):
            continue
        clean_count[word] = count

    # bigram 统计
    bigram = {}
    for i in range(len(all_readed_tokens) - 1):
        current_word = all_readed_tokens[i]
        next_word = all_readed_tokens[i + 1]
        if current_word not in clean_count or next_word not in clean_count:
            continue
        if current_word not in bigram:
            bigram[current_word] = {}
        if next_word not in bigram[current_word]:
            bigram[current_word][next_word] = 0
        bigram[current_word][next_word] += 1

    print(f"清洗前：{len(tokens_count)} 个词")
    print(f"清洗后：{len(clean_count)} 个词")
    print(f"bigram 条目数：{len(bigram)}")

    return clean_count, bigram


if __name__ == "__main__":
    all_readed_tokens = read_books(folder_path, files)
    clean_count, bigram = build_vocab_and_bigram(all_readed_tokens)
    print(f"共计{len(clean_count)}个不同的词")
