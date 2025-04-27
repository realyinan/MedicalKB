from torch.utils.data import DataLoader, Dataset
from process import *


class MyDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.datas = [json.loads(line) for line in open(data_path, "r", encoding="utf-8")]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        content = self.datas[item]  # 根据索引取出一个样本, 字典样式
        text = content["text"]
        spo_list = content["spo_list"]
        return text, spo_list


def get_dataloader():
    train_dataset = MyDataset(conf.train_data_path)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=conf.batch_size, collate_fn=collate_fn)

    test_dataset = MyDataset(conf.test_data_path)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=conf.batch_size, collate_fn=collate_fn)

    dev_dataset = MyDataset(conf.dev_data_path)
    dev_dataloader = DataLoader(dataset=dev_dataset, shuffle=True, batch_size=conf.batch_size, collate_fn=collate_fn)

    return train_dataloader, test_dataloader, dev_dataloader


if __name__ == "__main__":
    train_dataloader, test_dataloader, dev_dataloader = get_dataloader()
    for inputs, labels in train_dataloader:
        print(inputs)
        print(labels)
        break

