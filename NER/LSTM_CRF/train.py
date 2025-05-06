import torch
import torch.nn as nn
import torch.optim as optim
from model.BiLSTM import *
from model.BiLSTM_CRF import *
from utils.data_loader import *
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
    # 开始训练
    start_time = time.time()
    if conf.model == "BiLSTM":
        f1_score = -1000
        for epoch in range(conf.epochs):
            # 指定模型训练
            model.train()
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
                    print(f"当前批次: {epoch+1}, 损失是: {bilstm_loss.detach().item():.3f}")
            precision, recall, f1, report = model2dev(dev_dataloader, model, criterion)
            if f1 > f1_score:
                f1_score = f1
                torch.save(model.state_dict(), f="./save_model/bilstm_best.pth")
                print(report)
        end_time = time.time()
        print(f"BiLSTM的训练的总时间{end_time-start_time:.3f}")
    elif conf.model == "BiLSTM_CRF":
        f1_score = -1000
        for epoch in range(conf.epochs):
            # 指定模型训练
            model.train()
            for index, (inputs, labels, mask) in enumerate(tqdm(train_datalaoder, desc="BiLSTM_CRF训练")):
                # 将数据送入GPU
                x = inputs.to(conf.device)  # [4, 78]
                labels = labels.to(conf.device)  # [4, 78]
                mask = mask.to(torch.bool).to(conf.device)  # [4, 78]
                loss = model.log_likelihood(x, labels, mask).mean()  # 计算损失值
                # 优化器梯度清零
                optimizer.zero_grad()
                # 后向传播:计算梯度
                loss.backward()
                # 梯度裁剪: 防止梯度爆炸 通过限制梯度的最大范数（norm）来稳定训练过程。
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                # 梯度更新:更新梯度
                optimizer.step()
                if index % 200 == 0:
                    print(f"当前批次: {epoch + 1}, 损失是: {loss.detach().item():.3f}")
            precision, recall, f1, report = model2dev(dev_dataloader, model)
            if f1 > f1_score:
                f1_score = f1
                torch.save(model.state_dict(), f="./save_model/bilstm_crf_best.pth")
                print(report)
        end_time = time.time()
        print(f"BiLSTM_CRF的训练的总时间{end_time - start_time:.3f}")


def model2dev(dev_iter, model, criterion=None):
    """
    :param dev_iter: 验证集
    :param model: 选择的模型
    :param criterion: 损失函数对象
    :return:
    """
    av_loss = 0
    preds, golds = [], []
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
        elif model.name == "BiLSTM_CRF":
            mask = mask.to(torch.bool)
            predict  = model(val_x, mask)
            loss = model.log_likelihood(val_x, val_y, mask).mean()
            av_loss += loss.detach().item()

        # 获取真实的样本句子长度
        length = []
        for value in val_x.cpu():
            temp = []
            for j in value:
                if j.item() > 0:
                    temp.append(j)
            length.append(temp)

        # 提取真实样本句子长度的预测结果
        for idx, value in enumerate(predict):
            # value: 每个样本的预测结果
            preds.extend(value[:len(length[idx])])

        # 提取真实样本句子长度的真实结果
        for idx, value in enumerate(val_y.tolist()):
            # value: 每个样本的预测结果
            golds.extend(value[:len(length[idx])])

    # 计算一下平均损失
    av_loss = av_loss / len(dev_iter)
    # 计算指标
    precision = precision_score(golds, preds, average="weighted")
    recall = recall_score(golds, preds, average="weighted")
    f1 = f1_score(golds, preds, average="weighted")
    report = classification_report(golds, preds)
    return precision, recall, f1, report


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
    model2train()
    # test_dev()

