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
        best_triple_f1 = train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch)
    torch.save(model.state_dict(), "../save_model/last_model.pth")


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
        if index % 500 == 0 and index != 0:
            torch.save(model.state_dict(), f="../save_model/epoch_%s_step_%s.pth" %(epoch+1, index))
            results = model2dev(model, dev_iter)
            print(results[-1])
            if results[-2] >= best_triple_f1:
                best_triple_f1 = results[-2]
                torch.save(model.state_dict(), "../save_model/best_f1.pth")
                print('epoch:{},'
                      'index:{},'
                      'sub_precision:{:.4f}, '
                      'sub_recall:{:.4f}, '
                      'sub_f1:{:.4f}, '
                      'obj_precision:{:.4f}, '
                      'obj_recall:{:.4f}, '
                      'obj_f1:{:.4f},'
                      'train loss:{:.4f}'.format(epoch,
                                                 index,
                                                 results[0],
                                                 results[1],
                                                 results[2],
                                                 results[3],
                                                 results[4],
                                                 results[5],
                                                 loss.detach().item()))
    return best_triple_f1


def model2dev(model, dev_iter):
    model.eval()
    # 创建一个Dataframe对象: 存储指标
    df = pd.DataFrame(data=[[0, 0, 0, 0.0, 0.0, 0.0], [0, 0, 0, 0.0, 0.0, 0.0]], columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
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
            # 抽取预测的主实体
            pred_subs = extract_sub(pred_sub_heads[batch_idx].squeeze(), pred_sub_tails[batch_idx].squeeze())
            # 抽取真实的主实体
            true_subs = extract_sub(sub_heads[batch_idx].squeeze(), sub_tails[batch_idx].squeeze())
            # 抽取预测的客实体和关系
            pred_objs = extract_obj_and_rel(pred_obj_heads[batch_idx], pred_obj_tails[batch_idx])
            # 抽取真实的客实体和关系
            true_objs = extract_obj_and_rel(obj_heads[batch_idx], obj_tails[batch_idx])

            # 计算预测的主实体个数
            df.loc["sub", "PRED"] = len(pred_subs) + df.loc["sub", "PRED"]
            # 计算获取的主实体个数
            df.loc["sub", "REAL"] = len(true_subs) + df.loc["sub", "REAL"]

            for true_sub in true_subs:
                if true_sub in pred_subs:
                    df.loc["sub", "TP"] += 1

            # 计算预测的客实体及关系个数
            df.loc["triple", "PRED"] = len(pred_objs) + df.loc["triple", "PRED"]
            # 计算真实的客实体及关系个数
            df.loc["triple", "REAL"] = len(true_objs) + df.loc["triple", "REAL"]

            for true_obj in true_objs:
                if true_obj in pred_objs:
                    df.loc["triple", "TP"] += 1

    # 计算主实体精确率
    df.loc["sub", "p"] = df.loc["sub", "TP"] / (df.loc["sub", "PRED"] + 1e-9)
    # 计算主实体召回率
    df.loc["sub", "r"] = df.loc["sub", "TP"] / (df.loc["sub", "REAL"] + 1e-9)
    # 计算主实体的f1
    df.loc["sub", "f1"] = 2 * df.loc["sub", "p"] * df.loc["sub", "r"] / (df.loc["sub", "p"] + df.loc["sub", "r"] + 1e-9)

    # 计算客实体精确率
    df.loc["triple", "p"] = df.loc["triple", "TP"] / (df.loc["triple", "PRED"] + 1e-9)
    # 计算客实体召回率
    df.loc["triple", "r"] = df.loc["triple", "TP"] / (df.loc["triple", "REAL"] + 1e-9)
    # 计算客实体的f1
    df.loc["triple", "f1"] = 2 * df.loc["triple", "p"] * df.loc["triple", "r"] / (df.loc["triple", "p"] + df.loc["triple", "r"] + 1e-9)

    sub_precision = df.loc["sub", "p"]
    sub_recall = df.loc["sub", "r"]
    sub_f1 = df.loc["sub", "f1"]
    obj_precision = df.loc["triple", "p"]
    obj_recall = df.loc["triple", "r"]
    obj_f1 = df.loc["triple", "f1"]

    return sub_precision, sub_recall, sub_f1, obj_precision, obj_recall, obj_f1, df
    

if __name__ == '__main__':
    model, optimizer, sheduler, conf.device = load_model(conf)
    train_iter, test_iter, dev_iter = get_dataloader()
    model2train(model, train_iter, dev_iter, optimizer, conf)
