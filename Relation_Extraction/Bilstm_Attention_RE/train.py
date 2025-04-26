from bilstm_atten import *
from data_loader import *
from process import *
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm


def model2train(conf, vocab_size, pos_size, tag_size):
    # 获取数据
    train_dataloader, test_dataloader = get_data_loader()
    # 实例化模型
    model = BiLSTM_Atten(conf, vocab_size, pos_size, tag_size).to(conf.device)
    # 实例化优化器
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)
    # 实例化损失函数
    crossentropy = nn.CrossEntropyLoss()
    # 实现模型训练模式
    model.train()

    start_time = time.time()
    train_loss = 0.0
    train_acc = 0
    total_iter_num = 0
    total_sample = 0

    for epoch in range(conf.epochs):
        for index, (inputs, positionE1, positionE2, labels, _, _, _,) in enumerate(tqdm(train_dataloader, desc="开始训练")):
            out = model(inputs, positionE1, positionE2)
            loss = crossentropy(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印日志
            train_loss += loss.detach().item()
            # 预测正确的个数
            train_acc = train_acc + sum(torch.argmax(out, dim=-1) == labels).item()
            # 已经训练的样本个数
            total_sample = total_sample + labels.size(0)
            # 训练迭代次数
            total_iter_num += 1
            if total_iter_num % 25 == 0:
                avg_loss = train_loss / total_iter_num
                avg_acc = train_acc / total_sample
                use_time = time.time() - start_time
                print("轮次: %d, 损失: %.6f, 时间: %d, 准确率: %.3f" % (epoch+1, avg_loss, use_time, avg_acc))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f="./save_model/ba_model.pth")


if __name__ == "__main__":
    vocab_size = len(word2id)
    pos_size = 143
    tag_size = len(relation2id)
    model2train(conf, vocab_size, pos_size, tag_size)





