import os
from solt_config import *
conf = Config()

'''构造数据集'''


def build_data():
    datas = []
    sample_x = []
    sample_y = []
    vocab_list = ["PAD", 'UNK']
    with open(conf.train_path, 'r', encoding='utf-8') as fr:
        lines = fr.read().strip().split('\n\n')
    for line in lines:
        # print(line.split('\n'))
        for value in line.split('\n'):
            word, tag = value.split(' ')
            if word not in vocab_list:
                vocab_list.append(word)
            sample_x.append(word)
            sample_y.append(tag)
        datas.append([sample_x, sample_y])
        sample_x = []
        sample_y = []
    word2id = {wd: index for index, wd in enumerate(vocab_list)}
    write_file(vocab_list, conf.vocab_path)
    return datas, word2id


'''保存字典文件'''


def write_file(wordlist, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(wordlist))


if __name__ == '__main__':

    datas, word2id = build_data()
    print(len(datas))
    print(datas[:4])
    print(word2id)
    print(len(word2id))