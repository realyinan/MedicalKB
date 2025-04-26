from config import *
from itertools import chain


conf = Config()
relation2id = {}
with open(conf.rel_data_path, "r", encoding="utf-8") as fr:
    for line in fr.readlines():
        word, idx = line.strip().split(" ")
        if word not in relation2id:
            relation2id[word] = idx


def get_text_data(datapath):
    datas = []  # 数据
    labels = []  # 标签
    positionE1 = []  # 存储相对于实体一的位置信息
    positionE2 = []  # 存储相对于实体一的位置信息
    entities = []  # 存储实体信息
    rel2count = {key: 0 for key, value in relation2id.items()}  # 统计不同类别出现的次数
    with open(datapath, "r", encoding="utf-8") as f:
        for line in  f.readlines():
            line = line.rstrip().split(" ", maxsplit=3)
            if line[2] not in rel2count:
                continue
            # 确保样本均衡, 都不超过2000
            if rel2count[line[2]] > 3000:
                continue
            else:
                entities.append([line[0], line[1]])
                sentence = []
                index1 = line[3].index(line[0])
                position1= []
                index2 = line[3].index(line[1])
                position2 = []
                assert len(line) == 4
                for i, word in enumerate(line[3]):
                    sentence.append(word)
                    position1.append(i-index1)
                    position2.append(i-index2)
                datas.append(sentence)
                labels.append(relation2id[line[2]])
                positionE1.append(position1)
                positionE2.append(position2)
                rel2count[line[2]] += 1
    return datas, labels, positionE1, positionE2, entities


def get_word_id(datapath):
    datas, labels, positionE1, positionE2, entities = get_text_data(datapath)
    words = list(set(chain(*datas)))
    word2id = {word: idx for idx, word in enumerate(words)}
    id2word = {idx: word for idx, word in enumerate(words)}
    word2id["BLANK"] = len(word2id)
    word2id["UNKNOWN"] = len(word2id)
    id2word[len(id2word)] = "BLANK"
    id2word[len(id2word)] = "UNKNOWN"
    return word2id, id2word


def sent_padding(words, word2id):
    """把句子words转换为id形式, 并自动补全为max_len长度"""
    ids = []
    for word in words:
        if word in word2id:
            ids.append(word2id[word])
        else:
            ids.append(word2id["UNKNOWN"])
    if len(ids) >= conf.max_len:
        return ids[:conf.max_len]
    ids.extend([word2id["BLANK"]] * (conf.max_len-len(ids)))
    return ids


def pos(num):
    """将实体位置信息进行转换"""
    if num < -70:
        return 0
    elif -70 <= num <= 70:
        return num + 70
    else:
        return 142


def position_padding(pos_ids):
    """将id进行数字转化防止为负数, 并进行截断"""
    ids = [pos(ids) for ids in pos_ids]
    if len(ids) >= conf.max_len:
        return ids[:conf.max_len]
    ids.extend([142] * (conf.max_len - len(ids)))
    return ids


if __name__ == "__main__":
    word2id, id2word = get_word_id(conf.train_data_path)
    print(len(word2id))
    print(len(id2word))

