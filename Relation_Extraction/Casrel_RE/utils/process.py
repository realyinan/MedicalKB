from config import *
import torch
from random import choice
from collections import defaultdict

conf = Config()


def find_head_idx(source, target):
    # source代表原始的句子的id, target代表句子中的实体id表示
    taget_len = len(target)
    for i in range(len(source)):
        if source[i: i+taget_len] == target:
            return i
    return -1


def create_label(inner_inputs_ids, inner_triples, seq_len):
    """获取每个样本的: 主实体长度, 主实体开始和结束位置张量表示, 客实体以及对应关系实现张量表示"""
    inner_sub_heads, inner_sub_tails = torch.zeros(seq_len), torch.zeros(seq_len)
    inner_obj_heads = torch.zeros((seq_len, conf.num_rel))
    inner_obj_tails = torch.zeros((seq_len, conf.num_rel))
    innner_sub_head2tail = torch.zeros(seq_len)  # 随机抽取一个实体, 从开头一个词到末尾词的索引

    # 因为数据预处理代码还待优化,会有不存在关系三元组的情况，
    # 初始化一个主词的长度为1，即没有主词默认主词长度为1，
    # 防止零除报错,初始化任何非零数字都可以，没有主词分子是全零矩阵
    inner_sub_len = torch.tensor([1], dtype=torch.float)

    # 主体 到 客体以及关系 的 映射
    s2ro_map = defaultdict(list)
    for inner_triple in inner_triples:
        # 对每一个样本中的spo三元组进行数字化表示
        sub1 = conf.tokenizer(inner_triple["subject"], add_special_tokens=False)["input_ids"]
        obj1 = conf.tokenizer(inner_triple["object"], add_special_tokens=False)["input_ids"]
        rel1 = conf.rel_vocab.to_index(inner_triple["predicate"])
        inner_triple = (sub1, obj1, rel1)  # 编码后的inner_triple

        # 分别获取主客实体的开始索引位置
        sub_head_idx = find_head_idx(inner_inputs_ids, inner_triple[0])
        obj_head_idx = find_head_idx(inner_inputs_ids, inner_triple[1])
        if sub_head_idx != -1 and obj_head_idx != -1:
            sub = (sub_head_idx, sub_head_idx+len(inner_triple[0])-1)
            s2ro_map[sub].append((obj_head_idx, obj_head_idx+len(inner_triple[1])-1, inner_triple[2]))

    if s2ro_map:
        for s in s2ro_map:
            # s代表主实体
            inner_sub_heads[s[0]] = 1
            inner_sub_tails[s[1]] = 1
        # 随机选择其中的一个主体
        sub_head_idx, sub_tail_index = choice(list(s2ro_map.keys()))
        innner_sub_head2tail[sub_head_idx: sub_tail_index+1] = 1
        inner_sub_len = torch.tensor([sub_tail_index-sub_head_idx+1], dtype=torch.float)

        for obj in s2ro_map.get((sub_head_idx, sub_tail_index), []):
            inner_obj_heads[obj[0]][obj[2]] = 1
            inner_obj_tails[obj[1]][obj[2]] = 1
    return inner_sub_len, innner_sub_head2tail, inner_sub_heads, inner_sub_tails, inner_obj_heads, inner_obj_tails


def collate_fn(datas):
    text_list = [data[0] for data in datas]
    triple = [data[1] for data in datas]
    # 对一个批次的文本进行编码, 按照最长句子进行补齐
    inputs = conf.tokenizer.batch_encode_plus(text_list, padding=True)
    # 获取一个批次的样本个数
    batch_size = len(inputs["input_ids"])
    # 获取样本编码之后的长度
    seq_len = len(inputs["input_ids"][0])
    sub_heads = []  # 存放主实体开始位置信息
    sub_tails = []  # 存放主实体结尾位置信息
    obj_heads = []  # 存放客实体开始位置信息
    obj_tails = []  # 存放客实体结尾位置信息
    sub_len = []
    sub_head2tail = []
    # 遍历每一个样本进行实体信息的转化
    for batch_index in range(batch_size):
        # 根据索引取出当前样本对应的编码后的结果
        inner_inputs_ids = inputs["input_ids"][batch_index]
        # 根据索引取出当前样本对应的spo三元组
        inner_triples = triple[batch_index]
        # 获取 每个 样本的: 主实体长度, 主实体开始和结束位置张量表示, 客实体以及对应关系实现张量表示
        results = create_label(inner_inputs_ids, inner_triples, seq_len)
        sub_len.append(results[0])
        sub_head2tail.append(results[1])
        sub_heads.append(results[2])
        sub_tails.append(results[3])
        obj_heads.append(results[4])
        obj_tails.append(results[5])
    input_ids = torch.tensor(inputs["input_ids"]).to(conf.device)
    mask = torch.tensor(inputs["attention_mask"]).to(conf.device)
    sub_heads = torch.stack(sub_heads).to(conf.device)
    sub_tails = torch.stack(sub_tails).to(conf.device)
    sub_head2tail = torch.stack(sub_head2tail).to(conf.device)
    obj_heads = torch.stack(obj_heads).to(conf.device)
    obj_tails = torch.stack(obj_tails).to(conf.device)
    sub_len = torch.stack(sub_len).to(conf.device)

    inputs = {
        "input_ids": input_ids,
        "mask": mask,
        "sub_head2tail": sub_head2tail,
        "sub_len": sub_len
    }

    labels = {
        "sub_heads": sub_heads,
        "sub_tails": sub_tails,
        "obj_heads": obj_heads,
        "obj_tails": obj_tails
    }

    return inputs, labels


def convert_score_to_zero_one(tensor):
    # 以0.5为阈值, 大于0.5的设置为1, 小于0.5的设置为0
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor


def extract_sub(sub_head, sub_tail):
    """
    :param sub_head: 主实体开始位置
    :param sub_tail: 主实体结束位置
    :return: 列表里面对应的所有实体
    """
    # 获取所有位置为1的索引
    heads = torch.arange(0, len(sub_head), device=conf.device)[sub_head == 1]
    tails = torch.arange(0, len(sub_tail), device=conf.device)[sub_tail == 1]
    # 存储所有的主实体(start, end)
    subs = []
    for head, tail in zip(heads, tails):
        if tail >= head:
            subs.append((head.item(), tail.item()))
    return subs


def extract_obj_and_rel(obj_heads, obj_tails):
    """
    :param obj_heads: 客实体开始位置以及关系类型 [80, 18]
    :param obj_tails: 客实体结束位置以及关系类型 [80, 18]
    :return: 元素形状(rel_index, start_index, end_index)
    """
    obj_heads = obj_heads.T  # [18, 80]
    obj_tails= obj_tails.T  # [18, 80]
    rel_count = obj_heads.shape[0]
    obj_and_rels = []
    for rel_index in range(rel_count):
        obj_head = obj_heads[rel_index]
        obj_tail = obj_tails[rel_index]
        objs = extract_sub(obj_head, obj_tail)
        if objs:
            for obj in objs:
                start_index, end_index = obj
                obj_and_rels.append((rel_index, start_index, end_index))
    return obj_and_rels




