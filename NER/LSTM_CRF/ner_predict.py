import torch
import torch.nn as nn
import torch.optim as optim
from model.BiLSTM_CRF import *
from model.BiLSTM import *
from utils.data_loader import *

models = {
    "BiLSTM": NERBiLSTM, "BiLSTM_CRF": NERBiLSTMCRF
}
if conf.model == "BiLSTM":
    model = models[conf.model](conf.embedding_dim, conf.hidden_dim, conf.dropout, word2id, conf.tag2id)
    model.load_state_dict(torch.load("./save_model/bilstm_best.pth"))
elif conf.model == "BiLSTM_CRF":
    model = models[conf.model](conf.embedding_dim, conf.hidden_dim, conf.dropout, word2id, conf.tag2id)
    model.load_state_dict(torch.load("./save_model/bilstm_crf_best.pth"))

id2tag = {value: key for key, value in conf.tag2id.items()}
print(id2tag)


def model2test(sample: str):
    # 输入一个样本, 直接提取出样本中的实体
    x = []
    for char in sample:
        if char not in word2id:
            char = "UNK"
        x.append(word2id[char])
    x_train = torch.tensor([x])  # [1, 27]
    mask = (x_train != 0).long()
    model.eval()
    with torch.no_grad():
        if model.name == "BiLSTM":
            outputs = model(x_train, mask)  # [1, 27, 11]
            predict_ids = torch.argmax(outputs, dim=-1)[0]  # [1, 27]  [27]
            tags = [id2tag[i.item()] for i in predict_ids]
        else:
            predicts = model(x_train, mask)[0]
            tags = [id2tag[i] for i in predicts]
        chars = [i for i in sample]
        print(chars)
        print(tags)
        assert len(chars) == len(tags)
        result = extract_entities(chars, tags)
        return result


def extract_entities(tokens, tags):
    entities = []
    entity = []
    entity_type = None

    for token, label in zip(tokens, tags):
        if label.startswith("B-"):  # 代表实体的开始
            if entity:  # 保存上一个实体  冠心病糖尿病 这种特殊情况
                entities.append((entity_type, "".join(entity)))
                entity = []
            entity_type = label.split("-")[1]
            entity.append(token)
        elif label.startswith("I-") and entity:  # 代表实体的中间或者结束
            entity.append(token)
        else:
            if entity:  # 保存上一个实体
                entities.append((entity_type, "".join(entity)))
                entity = []
                entity_type = None
    if entity:
        entities.append((entity_type, "".join(entity)))
    return {entity: entity_type  for entity_type, entity in entities}


if __name__ == "__main__":
    result = model2test("小明的父亲患有糖尿病冠心病，无手术外伤史及药物过敏史")
    print(result)

