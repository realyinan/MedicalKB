import pandas as pd
import numpy as np
import os
import collections
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rich import print


entity_data = pd.read_csv("./entity_disambiguation/entity_list.csv", encoding="utf-8")
valid_data = pd.read_csv("./entity_disambiguation/valid_data.csv", encoding="GB2312")

# 将实体名称拼接成一个长字符串, 并用 "|" 分隔, 用于统计实体名称出现的次数
s = ''
keywortd_list = []
for i in entity_data["entity_name"].values.tolist():
    s += i + "|"
# print(s)

# 统计实体名称在字符串中的出现次数，如果某个名称出现次数超过一次，则将其加入keyword_list（关键词列表）
for k, v in collections.Counter(s.split("|")).items():
    if v > 1:
        keywortd_list.append(k)
# print(keywortd_list)

train_sentence = []
for i in entity_data["desc"].values:
    train_sentence.append(" ".join(jieba.lcut(i)))

vectorizer = TfidfVectorizer()

# 将实体描述信息转化为TF-IDF特征矩阵
X = vectorizer.fit_transform(train_sentence)
# print(f"当前特征词的总数: {vectorizer.get_feature_names_out()}")
# print(X)
# print(X.toarray().shape)

def get_entity(sentence):
    start_id = 1001
    # 对句子进行分词
    alist = [" ".join(jieba.lcut(sentence))]
    # 对上述语料进行TF-IDF的向量转换
    temp_vector = vectorizer.transform(alist)
    result = cosine_similarity(temp_vector, X)[0]
    top_idx = np.argsort(result)[-1]
    return top_idx + start_id


row = 0
result_data = []
neighbor_sentence = ""

for sentence in valid_data["sentence"]:
    res = [row]  # 初始化结果列表. 首先添加当前行号
    for key_word in keywortd_list:
        if key_word in sentence:
            k_len = len(key_word)
            ss = ""
            for i in range(len(sentence)-k_len+1):
                if sentence[i:i+k_len] == key_word:
                    s = str(i) + '-' + str(i+k_len) + ":"
                    # 获取包含关键词的邻近句子，用于计算实体相似度
                    if i > 10 and i + k_len < len(sentence) - 9:
                        neighbor_sentence = sentence[i - 10:i + k_len + 9]
                    elif i < 10:
                        neighbor_sentence = sentence[:20]
                    elif i + k_len > len(sentence) - 9:
                        neighbor_sentence = sentence[-20:]

                    # 调用get_entity函数, 获取与邻近句子最相似的实体ID
                    s += str(get_entity(neighbor_sentence))
                    ss += s + "|"
            res.append(ss[:-1])
    result_data.append(res)
    row += 1

print(result_data)



# get_entity("很多的网友都会有这样的一个问题,明明华盛顿和乾隆死在了同一年,为何却给人的感觉不是一个时代的人呢?")


