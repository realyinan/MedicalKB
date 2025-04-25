import jieba.posseg as pseg


# 需要进行关系抽取的样本数据
samples = ["2014年1月8日，杨幂与刘恺威的婚礼在印度尼西亚巴厘岛举行",
           "周星驰和吴孟达在《逃学威龙》中合作出演",
           '成龙出演了《警察故事》等多部经典电影']
# 定义需要抽取的关系集合
relations2dict = {'夫妻关系':['结婚', '领证', '婚礼'],
                  '合作关系': ['搭档', '合作', '签约'],
                  '演员关系': ['出演', '角色', '主演']}
# 继续实体和关系的抽取
for sample in samples:
    entities = []
    relations = []
    movie_name = []
    # 进行词性标注
    for word, flag in pseg.lcut(sample):
        if flag == "nr":  # nr 人名
            entities.append(word)
        elif flag == "x":  # x 非语素词
            if not movie_name:
                movie_name.append(sample.index(word))
            else:
                movie_name.append(sample.index(word))
                entities.append(sample[movie_name[0]+1: movie_name[1]])
        else:
            for key, value in relations2dict.items():
                if word in value:
                    relations.append(key)
    if len(entities) >= 2 and len(relations) >= 1:
        print(f"当前的样本: {sample}")
        print(f"抽取出来的结果: {entities[0]}->{relations[0]}->{entities[1]}")
    else:
        print("当前样本没有抽取出spo三元组")

