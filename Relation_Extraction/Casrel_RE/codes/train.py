import torch
from CasrelModel import *
from data_loader import *
from process import *
from config import *
from tqdm import tqdm
import pandas as pd

conf = Config()

def model2train(model, train_iter, dev_iter, optimizer, conf):
    # 定义初始f1值为0
    best_triple_f1 = 0
    # 开始外部迭代
    for epoch in range(conf.epochs):
        train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch)
        break


# 内部数据迭代函数
def train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch):
    for index, (inputs, labels) in enumerate(tqdm(train_iter, desc="Casrel训练")):
        model.train()
        # 将数据送入模型得到预测结果
        preds = model(**inputs)
        # 将计算预测结果和真实标签结果的损失
        loss = model.compute_loss(**preds, **labels)
        # 梯度清零
        model.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度更新
        optimizer.step()
        # 每隔1500步进行模型的验证
        if index % 1500 == 0:
            torch.save(model.state_dict(), f="../save_model/epoch_%s_step_%s.pth" %(epoch+1, index))
            model2dev(model, dev_iter)
        break


def model2dev(model, dev_iter):
    model.eval()
    # 创建一个Dataframe对象: 存储指标
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
    df.fillna(0, inplace=True)
    for inputs, labels in tqdm(dev_iter, desc="Casrel验证"):
        logist = model(**inputs)
        pred_sub_heads = convert_score_to_zero_one(logist["pred_sub_heads"])
        pred_sub_tails = convert_score_to_zero_one(logist["pred_sub_tails"])
        sub_heads = convert_score_to_zero_one(labels["sub_heads"])
        sub_tails = convert_score_to_zero_one(labels["sub_tails"])
        obj_heads = convert_score_to_zero_one(labels['obj_heads'])
        obj_tails = convert_score_to_zero_one(labels['obj_tails'])
        pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
        pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])
        batch_size = inputs["input_ids"].shape[0]

        for batch_idx in range(batch_size):
            pred_subs = extract_sub(pred_sub_heads[batch_idx].squeeze(), pred_sub_tails[batch_size].squeeze())

        break

def test_train():
    model, optimizer, sheduler, conf.device = load_model(conf)
    train_iter, test_iter, dev_iter = get_dataloader()
    model2train(model, train_iter, dev_iter, optimizer, conf)




if __name__ == '__main__':
    test_train()
