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

    def get_objs_for_specific_sub(self, sub_head2tail, sub_len, encoded_output):
        """
        :param sub_head2tail: [4, 1, 80]
        :param sub_len: [4, 1]
        :param encoded_output: [4, 80, 768]
        :return:
        """
        # 将主实体的信息从encoder_output中筛选出来, 筛选出, 1个批次4个样本, 每个样本1个实体, 对应的768维度向量
        sub = torch.matmul(sub_head2tail, encoded_output)  # [4, 1, 768]
        # 平均上述sub信息
        sub_len = sub_len.unsqueeze(1)  # [4, 1, 1]
        sub = sub / sub_len
        # 融合原始的bert编码之后的结果
        encoded_text = encoded_output + sub  # [4, 80, 768] + [4, 1, 768] -> [4, 80, 768]
        # 预测出客实体开始位置和对应关系
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))  # [4, 80, 18]
        # 预测出客实体开始位置和对应关系
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))  # [4, 80, 18]
        return pred_obj_heads, pred_obj_tails

    def forward(self, input_ids, mask, sub_head2tail, sub_len):
        """
        :param input_ids: [batch_size, seq_len]  [4, 80]
        :param mask: [batch_size, seq_len]  [4, 80]
        :param sub_head2tail: [batch_size, seq_len]  [4, 80]
        :param sub_len: [batch_size, 1]  [4, 1]
        :return:
        """
        # 将原始文本进行编码
        encoded_output = self.get_encoded_text(input_ids, mask)  # [4, 80, 768]
        # 将编码之后的结果送入get_subs函数, 预测主实体开始和结束位置
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_output)  # [4, 80, 1], [4, 80, 1]
        # 将bert模型编码后的结果融合主实体信息, 进行客实体和对应关系的解码
        sub_head2tail = sub_head2tail.unsqueeze(1)  # [4, 1, 80]
        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(sub_head2tail, sub_len, encoded_output)  # [4, 80, 18], [4, 80, 18]
        result_dict = {
            "pred_sub_heads": pred_sub_heads,
            "pred_sub_tails": pred_sub_tails,
            "pred_obj_heads": pred_obj_heads,
            "pred_obj_tails": pred_obj_tails,
            "mask": mask
        }
        return result_dict

    def compute_loss(self, pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails, mask, sub_heads, sub_tails, obj_heads, obj_tails):
        """
        :param pred_sub_heads: [4, 80, 1]
        :param pred_sub_tails: [4, 80, 1]
        :param pred_obj_heads: [4, 80, 18]
        :param pred_obj_tails: [4, 80, 18]
        :param mask: [4, 80]
        :param sub_heads: [4, 80]
        :param sub_tails: [4, 80]
        :param obj_heads: [4, 80, 18]
        :param obj_tails: [4, 80, 18]
        :return:
        """
        # 获取关系类别总数
        rel_count = obj_heads.shape[-1]  # 18
        # 将mask进行升维
        rel_mask = mask.unsqueeze(-1).repeat(1, 1, rel_count)  # [4, 80, 18]
        # 计算主实体开始位置损失
        loss1 = self.loss(pred_sub_heads, sub_heads, mask)
        # 计算主实体结束位置损失
        loss2 = self.loss(pred_sub_tails, sub_tails, mask)
        # 计算客实体开始位置及关系损失
        loss3 = self.loss(pred_obj_heads, obj_heads, rel_mask)
        # 计算客实体结束位置及关系损失
        loss4 = self.loss(pred_obj_tails, obj_tails, rel_mask)
        return loss1 + loss2 + loss3 + loss4

    def loss(self, pred, gold, mask):
        pred = pred.squeeze(-1)  # [4, 80, 1] -> [4, 80]
        my_loss = nn.BCELoss(reduction="none")(pred, gold)  # [4, 80]
        my_loss = torch.sum(my_loss*mask) / torch.sum(mask)
        return my_loss


def load_model(conf):
    model = Casrel(conf)
    model.to(conf.device)
    param_optimizer = list(model.named_parameters())
    # no_decay中存放不进行权重衰减的参数{因为bert官方代码对这三项免于正则化}
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=conf.learning_rate, eps=10e-8)
    # 是否对bert进行warm_up
    sheduler = None
    return model, optimizer, sheduler, conf.device


if __name__ == '__main__':
    # model = Casrel(conf)
    # model.to(conf.device)
    # train_dataloader, test_dataloader, dev_dataloader = get_dataloader()
    # for inputs, labels in train_dataloader:
    #     result_dict = model(**inputs)
    #     my_loss = model.compute_loss(**result_dict, **labels)
    #     print(my_loss)
    #     break
    load_model(conf)


