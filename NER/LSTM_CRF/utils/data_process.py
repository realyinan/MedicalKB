import json
import os
os.chdir("..")
cur = os.getcwd()
print("当前数据处理默认工作目录: ", cur)


class TransferData(object):
    def __init__(self):
        self.labels_dict = json.load(open(os.path.join(cur, "data/labels.json"), encoding="utf-8"))
        self.seq_tag_dict = json.load(open(os.path.join(cur, "data/tag2id.json"), encoding="utf-8"))
        self.origin_path = os.path.join(cur, "data_origin")
        self.train_filepath = os.path.join(cur, "data/train.txt")

    def transfer(self):
        with open(self.train_filepath, "w", encoding="utf-8") as fr:
            # 接下来分析数据, 然后存储
            for root, dirs, files in os.walk(self.origin_path):
                # 遍历所有的文件
                for file in files:
                    file_path = os.path.join(root, file)
                    if "original" not in file_path:
                        continue
                    label_filepath = file_path.replace(".txtoriginal", "")
                    res_dict = self.read_label_text(label_filepath)
                    with open(file_path, "r", encoding="utf-8") as fr1:
                        content = fr1.read().strip()
                    for idx, char in enumerate(content):
                        char_tag = res_dict.get(idx, "O")
                        str1 = char + "\t" + char_tag + "\n"
                        fr.write(str1)

    def read_label_text(self, path):
        res_dict = {}
        with open(path, "r", encoding="utf-8") as fr2:
            for line in fr2.readlines():
                res = line.strip().split("\t")
                start = int(res[1])
                end = int(res[2])
                label = res[3]
                label_tag = self.labels_dict.get(label)
                for i in range(start, end+1):
                    if i == start:
                        tag = "B-" + label_tag
                    else:
                        tag = "I-" + label_tag
                    res_dict[i] = tag
        return res_dict

if __name__ == "__main__":
    td = TransferData()
    # print(td.dict_labels)
    # print(td.seq_tag_dict)
    # print(td.origin_path)
    # print(td.train_filepath)
    print(td.transfer())
