import torch
from CasrelModel import *
from process import *
from config import *
from rich import print

conf = Config()

def create_model(conf):
    model = Casrel(conf)
    model.load_state_dict(torch.load("../save_model/best_f1.pth"))
    model.to(conf.device)
    return model


def get_inputs(sample, model):
    inputs = conf.tokenizer(sample)
    input_ids = torch.tensor([inputs["input_ids"]], device=conf.device)  # [1, 13]
    attention_mask = torch.tensor([inputs["attention_mask"]], device=conf.device)  # [1, 13]

    seq_len = len(inputs["input_ids"])
    inner_sub_head2tail = torch.zeros(seq_len)
    inner_sub_len = torch.tensor([1], dtype=torch.float)

    # 获取模型预测的主实体位置信息
    model.eval()
    with torch.no_grad():
        bert_encodced= model.get_encoded_text(input_ids, attention_mask)  # 获取bert编码之后的结果  [1, 13, 768]
        # 根据编码结果获取模型预测的主实体的开始和结束位置分数
        pred_sub_heads, pred_sub_tails = model.get_subs(bert_encodced)  # [1, 13, 1], [1, 13, 1]
        pred_sub_tails = convert_score_to_zero_one(pred_sub_tails)
        pred_sub_heads = convert_score_to_zero_one(pred_sub_heads)
        pred_subs = extract_sub(pred_sub_heads.squeeze(), pred_sub_tails.squeeze())

        if len(pred_subs) != 0:
            sub_head_idx = pred_subs[0][0]
            sub_tail_idx = pred_subs[0][1]

            # 获取主体长度以及主体位置全部赋值为1
            inner_sub_head2tail[sub_head_idx: sub_tail_idx+1] = 1
            inner_sub_len = torch.tensor([sub_tail_idx+1-sub_head_idx], dtype=torch.float)
    sub_len = inner_sub_len.unsqueeze(0).to(conf.device)  #[13, ]
    sub_head2tail = inner_sub_head2tail.unsqueeze(0).to(conf.device)  # [1, 13]

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
    logist = model(**inputs)
    pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])  # [1, 13, 1]
    pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])  # [1, 13, 1]
    pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])  # [1, 13, 18]
    pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])  # # [1, 13, 18]
    new_dict = {}
    spo_list = []
    ids = inputs['input_ids'][0]
    text_list = conf.tokenizer.convert_ids_to_tokens(ids)
    sentence = "".join(text_list[1:-1])
    pred_subs = extract_sub(pred_sub_heads.squeeze(), pred_sub_tails.squeeze())
    pred_objs = extract_obj_and_rel(pred_obj_heads.squeeze(), pred_obj_tails.squeeze())
    print(pred_subs)
    print(pred_objs)

    if len(pred_subs) == 0 or len(pred_objs) == 0:
        print("未识别出结果")
        return {}
    if len(pred_objs) > len(pred_subs):
        pred_subs = pred_subs * len(pred_objs)

    for sub, rel_obj in zip(pred_subs, pred_objs):
        sub_spo = {}
        sub_head, sub_tail = sub
        sub = "".join(text_list[sub_head: sub_tail+1])
        if "[PAD]" in sub:
            continue
        sub_spo["subject"] = sub

        relation = rel_id_word[str(rel_obj[0])]
        obj_head, obj_tail = rel_obj[1], rel_obj[2]
        obj = "".join(text_list[obj_head: obj_tail+1])
        if "[PAD]" in obj:
            continue
        sub_spo["predicate"] = relation
        sub_spo["object"] = obj
        spo_list.append(sub_spo)
    new_dict["text"] = sentence
    new_dict["spolist"] = spo_list
    return new_dict



if __name__ == '__main__':
    model = create_model(conf)
    sample = "林巧巧，女，汉族，1982年8月出生，福建厦门人，中共党员，2005年毕业于厦门大学新闻传播专业后即进入宣传系统工作，硕士学历。"
    result = model2predict(sample, model)
    print(result)