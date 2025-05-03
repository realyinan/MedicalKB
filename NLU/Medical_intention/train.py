import time
import torch
import torch.optim as optime
from model import *
from tqdm import tqdm
from utils.data_loader import *


def model2train():
    # 获取数据
    train_iter, test_iter = get_dataloader()
    # 实例化模型
    model = MyModel(conf.bert_path, conf.bert_hiden, conf.num_class)
    model.to(conf.device)
    # 实例化优化器
    my_optim = optime.Adam(model.parameters(), lr=conf.lr)
    # 实例化损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义模型训练参数
    model.train()
    total_num = 0  # 训练的样本个数
    total_loss = 0
    total_acc = 0

    # 开始训练
    start_time = time.time()
    for epoch_idx in range(conf.epochs):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(train_iter)):
            output = model(input_ids, attention_mask, token_type_ids)
            # 计算损失
            my_loss = criterion(output, labels)
            # 梯度清零
            my_optim.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            my_optim.step()

            total_loss = total_loss + my_loss.detach()
            total_num = total_num + output.shape[0]
            acc_num = sum(torch.argmax(output, dim=-1) == labels).item()
            total_acc = total_acc + acc_num
            if i % 100 == 0:
                avg_loss = total_loss / total_num
                avg_acc = total_acc / total_num
                usetime = time.time() - start_time
                print(f"当前训练轮次: {epoch_idx+1}, 平均损失: {avg_loss:.3f}, 平均准确率: {avg_acc:.2f}, 用时: {usetime}")
        torch.save(model.state_dict(), f="./save_model/epoch_%d.pth" %(epoch_idx+1))
    end_time = time.time()
    print(f"总耗时: {end_time-start_time}")


if __name__ == '__main__':
    model2train()
