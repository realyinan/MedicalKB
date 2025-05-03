import time
import torch
import torch.nn as nn
import torch.optim as optim
from model.BiLSTM_CRF import *
from utils.data_loader import *
from  tqdm import tqdm
# classification_report可以导出字典格式，修改参数：output_dict=True，可以将字典在保存为csv格式输出
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
conf = Config()

def model2train():
    # 获取数据
    train_dataloader, dev_dataloader = get_data()
    # 实例化模型
    models = {'BiLSTM_CRF': NERLSTM_CRF}
    model = models[conf.model](conf.embedding_dim, conf.hidden_dim, conf.dropout, word2id, conf.tag2id)
    model = model.to(conf.device)
    # 实例化损失函数
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(conf.MODEL.parameters(), lr=conf.LR, weight_decay=opt.weight_decay)
    # 实例化优化器
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    model.train()
    # 选择模型进行训练
    start_time = time.time()
    f1_score = -1000
    for epoch in range(conf.epochs):
        for index, (inputs, labels, mask)in enumerate(tqdm(train_dataloader, desc='bilstm+crf训练')):
            x = inputs.to(conf.device)
            mask = mask.to(torch.bool).to(conf.device)
            tags = labels.to(conf.device)
            # CRF
            loss = model.log_likelihood(x, tags, mask).mean()
            optimizer.zero_grad()
            loss.backward()
            # CRF
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
            optimizer.step()
            if index % 20 == 0:
                print('epoch:%04d,------------loss:%f' % (epoch, loss.item()))
        precision, recall, f1, report = model2dev(dev_dataloader, model)
        if f1 > f1_score:
            f1_score = f1
            torch.save(model.state_dict(), 'save_model/bilstm_crf_best.pth')
            print(report)
    end_time = time.time()
    print(f'训练总耗时：{end_time-start_time}')


def model2dev(dev_iter, model, criterion=None):
    aver_loss = 0
    preds, golds = [], []
    model.eval()
    for index, (inputs, labels, mask) in enumerate(tqdm(dev_iter, desc="测试集验证")):
        val_x = inputs.to(conf.device)
        mask = mask.to(conf.device)
        val_y = labels.to(conf.device)
        predict = []
        mask = mask.to(torch.bool)
        predict = model(val_x, mask)
        loss = model.log_likelihood(val_x, val_y, mask)
        aver_loss += loss.mean().item()
        # 统计非0的，也就是真实标签的长度
        leng = []
        for i in val_y.cpu():
            tmp = []
            for j in i:
                if j.item() > 0:
                    tmp.append(j.item())
            leng.append(tmp)
        # 提取真实长度的预测标签

        for index, i in enumerate(predict):
            preds.extend(i[:len(leng[index])])

        # 提取真实长度的真实标签
        for index, i in enumerate(val_y.tolist()):
            golds.extend(i[:len(leng[index])])
    aver_loss /= (len(dev_iter) * 64)
    precision = precision_score(golds, preds, average='macro')
    recall = recall_score(golds, preds, average='macro')
    f1 = f1_score(golds, preds, average='macro')
    report = classification_report(golds, preds)
    return precision, recall, f1, report


if __name__ == '__main__':
    model2train()