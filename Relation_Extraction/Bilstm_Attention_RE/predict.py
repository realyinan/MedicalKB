from model.bilstm_atten import *
from utils.data_loader import *
from utils.process import *
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm


def model2dev():
    _, test_dataloader = get_data_loader()
    # 加载预训练好的模型
    vocab_size = len(word2id)
    pos_size = 143
    tag_size = len(relation2id)
    model = BiLSTM_Atten(conf, vocab_size, pos_size, tag_size).to(conf.device)
    model.load_state_dict(torch.load("./save_model/ba_model.pth"))
    id2relation = {int(value): key for key, value in relation2id.items()}
    model.eval()
    with torch.no_grad():
        for index, (inputs, positionE1, positionE2, tags, sequences, labels, entities) in enumerate(tqdm(test_dataloader, desc="开始评估")):
            # print(f"原始句子: {sequences}")
            # print(f"原始关系: {labels}")
            # print(f"原始实体: {entities}")
            outputs = model(inputs, positionE1, positionE2)
            predicts = torch.argmax(outputs, dim=-1).tolist()
            # print(f"预测关系{predicts}")

            for i in range(len(sequences)):
                original_sentence = "".join(sequences[i])
                original_labels = id2relation[labels[i]]
                original_entities = entities[i]
                pred_label = id2relation[predicts[i]]
                print(f"原始句子: {original_sentence}")
                print(f"原始关系类别: {original_labels}")
                print(f"实体列表: {original_entities}")
                print(f"预测的关系: {pred_label}")
                print("=======================================================================")


if __name__ == '__main__':
    model2dev()
