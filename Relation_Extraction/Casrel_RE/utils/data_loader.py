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


def test_dataset():
    ds = MyDataset(conf.train_data_path)
    print(len(ds))
    print(ds[0])
    print(ds[1])


if __name__ == "__main__":
    test_dataset()

