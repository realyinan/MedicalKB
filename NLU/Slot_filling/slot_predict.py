import torch.nn as nn
import torch.optim as optim
from model.BiLSTM_CRF import *
from tqdm import tqdm
from solt_config import *
conf = Config()

# 实例化模型
models = {'BiLSTM_CRF': NERLSTM_CRF}
model = models["BiLSTM_CRF"](conf.embedding_dim, conf.hidden_dim, conf.dropout, word2id, conf.tag2id)
model.load_state_dict(torch.load('save_model/bilstm_crf_best.pth', map_location=conf.device))

id2tag = {value: key for key, value in conf.tag2id.items()}

def model2test(sample):
    x = []
    for char in sample:
        if char not in word2id:
            char = "UNK"
        x.append(word2id[char])

    x_train = torch.tensor([x])
    mask = (x_train != 0).long()
    model.eval()
    with torch.no_grad():
        if model.name =="BiLSTM":
            outputs = model(x_train, mask)
            preds_ids = torch.argmax(outputs,dim=-1)[0]
            tags = [id2tag[i.item()] for i in preds_ids]
        else:
            preds_ids = model(x_train, mask)
            tags = [id2tag[i] for i in preds_ids[0]]
        chars = [i for i in sample]
        assert len(chars) == len(tags)
        result = extract_entities(chars, tags)
        return result



def extract_entities(tokens, labels):
    entities = []
    entity = []
    entity_type = None

    for token, label in zip(tokens, labels):
        if label.startswith("B_"):  # 实体的开始
            if entity:  # 如果已经有实体，先保存
                entities.append((entity_type, ''.join(entity)))
                entity = []
            entity_type = label.split('_')[1]
            entity.append(token)
        elif label.startswith("I_") and entity:  # 实体的中间或结尾
            entity.append(token)
        else:
            if entity:  # 保存上一个实体
                entities.append((entity_type, ''.join(entity)))
                entity = []
                entity_type = None

    # 如果最后一个实体没有保存，手动保存
    if entity:
        entities.append((entity_type, ''.join(entity)))

    return {entity: entity_type for entity_type, entity in entities}


if __name__ == '__main__':
    result = model2test(sample='最近身体总是感觉浑身无力，发热，这是什么原因')
    print(result)

