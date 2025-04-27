import torch
import torch.nn as nn
from transformers import BertModel
from torch.optim import AdamW
from config import *
from data_loader import *
conf = Config()


class Casrel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # 定义预训练模型层
        self.bert = BertModel.from_pretrained(conf.bert_path)
        # 定义第一个全连接层: 识别主实体开始的位置
        self.sub_heads_linear = nn.Linear(conf.bert_dim, 1)
        # 定义第二个全连接层: 识别主实体结束的位置
        self.sub_tails_linear = nn.Linear(conf.bert_dim, 1)
        # 定义第三个全连接层: 识别客实体开始的位置以及关系类型
        self.obj_heads_linear = nn.Linear(conf.bert_dim, conf.num_rel)
        # 定义第四个全连接层: 识别客实体结束的位置以及关系类型
        self.obj_tails_linear = nn.Linear(conf.bert_dim, conf.num_rel)

    def get_encoded_text(self, input_ids, mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=mask)["last_hidden_state"]
        return bert_output

    def get_subs(self, encoder_text):
        # 预测主实体的开始位置
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoder_text))
        # 预测主实体的结束位置
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoder_text))
        return pred_sub_heads, pred_sub_tails


    def forward(self, input_ids, mask, sub_head2tail, sub_len):
        """
        :param input_ids: [batch_size, seq_len]  [4, 80]
        :param mask: [batch_size, seq_len]  [4, 80]
        :param sub_head2tail: [batch_size, seq_len]  [4, 80]
        :param sub_len: [batch_size, 1]  [4, 80]
        :return:
        """
        # 将原始文本进行编码
        encoded_output = self.get_encoded_text(input_ids, mask)  # [4, 80, 768]
        # 将编码之后的结果送入get_subs函数, 预测主实体开始和结束位置
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_output)  # [4, 80, 1], [4, 80, 1]
        print(pred_sub_heads.shape)
        print(pred_sub_tails.shape)



if __name__ == '__main__':
    model = Casrel(conf)
    model.to(conf.device)
    train_dataloader, test_dataloader, dev_dataloader = get_dataloader()
    for inputs, labels in train_dataloader:
        model(**inputs)
        break


