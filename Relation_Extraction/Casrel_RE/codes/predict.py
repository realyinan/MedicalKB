import torch
from CasrelModel import *
from process import *
from config import *

conf = Config()

def create_model(conf):
    model = Casrel(conf)
    model.load_state_dict(torch.load("../save_model/best_f1.pth"))
    model.to(conf.device)
    return model


def get_inputs(sample, model):
    inputs = conf.tokenizer(sample)
    input_ids = torch.tensor([inputs["input_ids"]], device=conf.device)
    attention_mask = torch.tensor([inputs["attention_mask"]], device=conf.device)

    seq_len = len(inputs["input_ids"])
    inner_sub_head2tail = torch.zeros(seq_len)
    inner_sub_len = torch.tensor([1], dtype=torch.float)

    # 获取模型预测的主实体位置信息
    model.eval()
    with torch.no_grad():
        bert_encodced= model.get_encoded_text(input_ids, attention_mask)  # 获取bert编码之后的结果  [1, 13, 768]
        # 根据编码结果获取模型预测的主实体的开始和结束位置分数
        pred_sub_heads, pred_sub_tails = model.get_subs(bert_encodced)
        pred_sub_tails = convert_score_to_zero_one(pred_sub_tails)
        pred_sub_heads = convert_score_to_zero_one(pred_sub_heads)
        pred_subs = extract_sub(pred_sub_heads.squeeze(), pred_sub_tails.squeeze())

        if len(pred_subs) != 0:
            sub_head_idx = pred_subs[0][0]
            sub_tail_idx = pred_subs[0][1]

            # 获取主体长度以及主体位置全部赋值为1
            inner_sub_head2tail[sub_head_idx: sub_tail_idx+1] = 1
            inner_sub_len = torch.tensor([sub_tail_idx+1-sub_head_idx], dtype=torch.float)
    sub_len = inner_sub_len.unsqueeze(0).to(conf.device)
    sub_head2tail = inner_sub_head2tail.unsqueeze(0).to(conf.device)

    inputs = {
        "input_ids": input_ids,
        "mask": attention_mask,
        "sub_head2tail": sub_head2tail,
        "sub_len": sub_len
    }

    return inputs, model

def model2predict(sample, model):
    with open(conf.rel_dict_path, "r", encoding="utf-8") as f:
        rel_id_word = json.load(f)
    # 获取模型的输入
    inputs, model = get_inputs(sample, model)
    print(inputs)
    print(model)



if __name__ == '__main__':
    model = create_model(conf)
    sample = "《七里香》是周杰伦演唱歌曲。"
    model2predict(sample, model)