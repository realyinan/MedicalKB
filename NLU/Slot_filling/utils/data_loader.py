import json
import torch
import os
print(os.getcwd())
from utils.common import *
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

datas, word2id = build_data()
print(len(datas))

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
    x_train = [torch.tensor([word2id[char] for char in data[0]]) for data in batch]
    y_train = [torch.tensor([conf.tag2id[label] for label in data[1]]) for data in batch]
    # 补齐input_ids, 使用0作为填充值
    input_ids_padded = pad_sequence(x_train, batch_first=True, padding_value=0)
    # # 补齐labels，使用0作为填充值
    labels_padded = pad_sequence(y_train, batch_first=True, padding_value=0)

    # 创建attention mask
    attention_mask = (input_ids_padded != 0).long()
    return input_ids_padded, labels_padded, attention_mask


def get_data():

    train_dataset = NerDataset(datas[:1800])
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=conf.batch_size,
                                  collate_fn=collate_fn,
                                  drop_last=True,
                                  )

    dev_dataset = NerDataset(datas[1800:])
    dev_dataloader = DataLoader(dataset=dev_dataset,
                                batch_size=conf.batch_size,
                                collate_fn=collate_fn,
                                drop_last=True,
                                )
    return train_dataloader, dev_dataloader

if __name__ == '__main__':
    train_dataloader, dev_dataloader = get_data()
    print(f'len(train_dataloader)-->{len(train_dataloader)}')
    print(f'len(dev_dataloader)-->{len(dev_dataloader)}')
    for input_ids_padded, labels_padded, attention_mask in train_dataloader:
        print(input_ids_padded.shape)
        print(labels_padded.shape)
        print(attention_mask.shape)
        break