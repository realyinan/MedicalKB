import torch
from common import *
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence  # 进行句子的长短补齐或者截断


datas, word2id = build_data()

class NerDataset(Dataset):
    def __init__(self, datas):
        super().__init__()
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        x = self.datas[item][0]
        y = self.datas[item][1]
        return x, y


def collate_fn(batch):
    x_train = [torch.tensor([word2id.get(word) for word in data[0]]) for data in batch]
    y_train = [torch.tensor([conf.tag2id.get(tag) for tag in data[1]]) for data in batch]
    # pad_sequence可以对一个批次的样本进行统一长度, 最后的长度是以该批次中最长的样本为基准
    input_ids_padded = pad_sequence(x_train, batch_first=True, padding_value=0)
    # 对labels进行补齐, 一般用-100
    labels_padded = pad_sequence(y_train, batch_first=True, padding_value=0)
    # 创建attention_mask
    attention_mask = (input_ids_padded != 0).long()
    return input_ids_padded, labels_padded, attention_mask

def get_data():
    """
    获取dataloader数据迭代器
    :return:
    """
    train_dataset = NerDataset(datas[:6200])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=conf.batch_size, collate_fn=collate_fn, drop_last=True, shuffle=True)

    dev_dataset = NerDataset(datas[6200:])
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=conf.batch_size, collate_fn=collate_fn, drop_last=True, shuffle=True)
    return  train_dataloader, dev_dataloader


if __name__ == "__main__":
    train_dataloader, dev_dataloader = get_data()
    for input_ids_padded, labels_padded, attention_mask in train_dataloader:
        print(input_ids_padded.shape)
        print(labels_padded.shape)
        print(attention_mask.shape)
        break


