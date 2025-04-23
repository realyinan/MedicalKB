import jieba.posseg as pseg
import re


org_tag = ['公司', '有限公司', '大学', '政府', '人民政府', '总局']

def excert_org(text):
    """
    :param text: 需要抽取实体的文本, 属于字符串类型
    :return:
    """
    # 1. 对该文本进行词性标注
    words_flags = pseg.lcut(text)
    # print(words_flags)
    # 2. 定义两个空列表
    words = []
    features = []
    # 3. 遍历词性标注后的结果
    for word, flag in words_flags:
        words.append(word)
        if word in org_tag:
            features.append("E")
        else:
            if flag == "ns":
                features.append("B")
            else:
                features.append("O")
    # print(words)
    # print(features)
    labels = "".join(features)
    # print(labels)
    # 用正则表达式提取实体
    pattern = re.compile("B+O*E+")
    match_label = re.finditer(pattern, labels)  # 返回一个包含match对象的迭代器
    match_list = []
    for ne in match_label:
        entity = words[ne.start(): ne.end()]
        match_list.append("".join(entity))
    return match_list


if __name__ == "__main__":
    text = "可在接到本决定书之日起六十日内向中国国家市场监督管理总局申请行政复议,杭州海康威视数字技术股份有限公司."
    ner = excert_org(text)
    print(ner)