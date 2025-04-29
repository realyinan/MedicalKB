import torch
from CasrelModel import *
from data_loader import *
from process import *
from config import *
from tqdm import tqdm
import pandas as pd

conf = Config()


def model2test(model, test_iter):
    model.eval()
    # 创建一个Dataframe对象: 存储指标
    df = pd.DataFrame(data=[[0, 0, 0, 0.0, 0.0, 0.0], [0, 0, 0, 0.0, 0.0, 0.0]],
                      columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
    with torch.no_grad():
        for inputs, labels in tqdm(test_iter, desc="Casrel测试"):
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
        df.loc["triple", "f1"] = 2 * df.loc["triple", "p"] * df.loc["triple", "r"] / (
                    df.loc["triple", "p"] + df.loc["triple", "r"] + 1e-9)
    return df

if __name__ == '__main__':
    model = Casrel(conf)
    model.load_state_dict(torch.load("../save_model/epoch_2_step_500.pth"))
    model.to(conf.device)
    _, test_iter, _  = get_dataloader()
    df = model2test(model, test_iter)
    print(df)
