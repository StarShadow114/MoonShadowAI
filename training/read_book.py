import os
import jieba

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


tokens_count = {}
all_readed_tokens = read_books(folder_path, files)
for readed_tokens in all_readed_tokens:
    if readed_tokens in tokens_count:
        tokens_count[readed_tokens] += 1
    else:
        tokens_count[readed_tokens] = 1
print(f"共计{len(tokens_count)}个不同的词")
