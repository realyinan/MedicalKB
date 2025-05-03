import torch
import pandas as pd
from intent_config import *
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

conf = Config()
tokenizer = BertTokenizer.from_pretrained(conf.bert_path)


def load_data(path):
    data = pd.read_csv(path, sep=",")
    texts = data["text"].to_list()
    labels = data["label_id"].map(int).to_list()
    return texts, labels


class MyDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.texts, self.labels = load_data(data_path)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        return text, label


def collate_fn(datas):
    batch_text = [data[0] for data in datas]
    batch_label = [data[1] for data in datas]
    inputs = tokenizer.batch_encode_plus(batch_text, padding="max_length", truncation=True, max_length=conf.max_len, return_tensors="pt")
    input_ids = inputs["input_ids"].to(conf.device)
    attention_mask = inputs["attention_mask"].to(conf.device)
    token_type_ids = inputs["token_type_ids"].to(conf.device)
    labels = torch.tensor(batch_label, dtype=torch.long, device=conf.device)
    return input_ids, attention_mask, token_type_ids, labels


def get_dataloader():
    train_dataset = MyDataset(conf.train_path)
    train_iter = DataLoader(dataset=train_dataset, shuffle=True, batch_size=conf.batch_size, drop_last=True, collate_fn=collate_fn)

    test_dataset = MyDataset(conf.test_path)
    test_iter = DataLoader(dataset=test_dataset, shuffle=True, batch_size=conf.batch_size, drop_last=True, collate_fn=collate_fn)
    return train_iter, test_iter

if __name__ == "__main__":
    train_iter, test_iter = get_dataloader()
    for input_ids, attention_mask, token_type_ids, labels in test_iter:
        print(input_ids.shape)
        print(attention_mask.shape)
        print(token_type_ids.shape)
        print(labels.shape)
        break