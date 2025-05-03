import torch
import torch.nn as nn
from utils.data_loader import *
from transformers import BertModel


class MyModel(nn.Module):
    def __init__(self, bert_path, bert_hidden, tag_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.linear = nn.Linear(bert_hidden, tag_size)


    def forward(self, input_ids, attention_mask, token_type_ids):
        pooler_output = self.bert(input_ids, attention_mask, token_type_ids)["pooler_output"]
        output = self.linear(pooler_output)
        return output


if __name__ == "__main__":
    train_iter, test_iter = get_dataloader()
    model = MyModel(conf.bert_path, conf.bert_hiden, conf.num_class)
    model.to(conf.device)
    for input_ids, attention_mask, token_type_ids, labels in train_iter:
        output = model(input_ids, attention_mask, token_type_ids)
        print(output.shape)
        break
