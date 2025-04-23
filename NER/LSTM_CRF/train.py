import torch
import torch.nn as nn
import torch.optim as optim
from BiLSTM import *
from BiLSTM_CRF import *
from data_loader import *
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from config import *
from tqdm import tqdm
import time
conf = Config()


def model2train():
    # 获取数据
    train_datalaoder, dev_dataloader = get_data()
    # 实例化模型
    models = {
        "BiLSTM": NERBiLSTM, "BiLSTM_CRF" : NERBiLSTMCRF
    }
    # 根据config文件中指定的模型进行实例化
    model = models[conf.model](conf.embedding_dim, conf.hidden_dim, conf.dropout, word2id, conf.tag2id)
    model.to(conf.device)
    # 实例化损失函数对象
    criterion = nn.CrossEntropyLoss(reduction="mean")
    # 实例化优化器对象
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    # 指定模型训练
    model.train()
    # 开始训练
    start_time = time.time()
    if conf.model == "BiLSTM":
        for epoch in range(conf.epochs):
            for index, (inputs, labels, mask) in enumerate(tqdm(train_datalaoder, desc="BiLSTM训练")):
                # 将数据送入GPU
                x = inputs.to(conf.device)  # [4, 78]
                labels = labels.to(conf.device)  # [4, 78]
                mask = mask.to(conf.device)  # [4, 78]
                output = model(x, mask)  # [4, 78, 11]
                # 进行预测值和损失值的损失计算, 需要进行更改的数据的形状
                pred = output.view(-1, len(conf.tag2id))  # [4*78, 11]
                # 计算损失
                bilstm_loss= criterion(pred, labels.view(-1))
                # 优化器梯度清零
                optimizer.zero_grad()
                # 后向传播:计算梯度
                bilstm_loss.backward()
                # 梯度更新:更新梯度
                optimizer.step()
                if index % 200 == 0:
                    print(f"当前批次: {epoch+1}, 损失是: {bilstm_loss:.3f}")


def model2dev(dev_iter, model, criterion=None):
    """
    :param dev_iter: 验证集
    :param model: 选择的模型
    :param criterion: 损失函数对象
    :return:
    """
    av_loss = 0
    pred, gold = [], []
    model.eval()
    for index, (inputs, labels, mask) in enumerate(tqdm(dev_iter, desc="验证集验证")):
        val_x = inputs.to(conf.device)  # [4, 78]
        val_y = labels.to(conf.device)  # [4, 78]
        mask = mask.to(conf.device)  # [4, 78]
        predict = []
        if model.name == "BiLSTM":
            output = model(val_x, mask)  # [4, 78, 11]
            predict = torch.argmax(output, dim=-1).tolist()
            pred = output.view(-1, len(conf.tag2id))
            model_loss = criterion(pred, val_y.view(-1))
            av_loss += model_loss.detach().item()
            print(av_loss)
            break


def test_dev():
    # 获取数据
    train_datalaoder, dev_dataloader = get_data()
    # 实例化模型
    models = {
        "BiLSTM": NERBiLSTM, "BiLSTM_CRF": NERBiLSTMCRF
    }
    # 根据config文件中指定的模型进行实例化
    model = models[conf.model](conf.embedding_dim, conf.hidden_dim, conf.dropout, word2id, conf.tag2id)
    model.to(conf.device)
    # 实例化损失函数对象
    criterion = nn.CrossEntropyLoss(reduction="mean")
    model2dev(dev_dataloader, model, criterion)


if __name__ == "__main__":
    # model2train()
    test_dev()

