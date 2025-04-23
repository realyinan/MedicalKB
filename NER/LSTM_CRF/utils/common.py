from config import *
conf = Config()


def build_data():
    """
    构造数据集: 对train.txt进行分析处理, 得到x, y样本对, 以标点符号分开
    :return:
    """
    datas = []
    sample_x = []
    sample_y = []
    vocab_list = ["PAD", "UNK"]
    with open(conf.train_path, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            word_tag_list = line.strip().split("\t")
            if not word_tag_list:
                continue
            word = word_tag_list[0]
            if not word:
                continue
            tag = word_tag_list[-1]
            sample_x.append(word)
            sample_y.append(tag)
            if word not in vocab_list:
                vocab_list.append(word)
            if word in ['。', '?', '!', '！', '？']:
                datas.append([sample_x, sample_y])
                sample_x = []
                sample_y = []
    word2id = {wd: index for index, wd in enumerate(vocab_list)}
    write_file(vocab_list, conf.vocab_path)
    return datas, word2id



def write_file(vocab_list, file_path):
    """
    将词表写入文件中
    :return:
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab_list))


if __name__ == "__main__":
    datas, word2id = build_data()
    print(len(datas))
    print(datas[:2])
    print(word2id)




