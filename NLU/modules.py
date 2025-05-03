import os
import re
import requests
import random
from py2neo import Graph
from Chatty_intention.clf_model import CLFModel
from config import *
import json
graph = Graph("http://localhost:7474/", auth=("neo4j", "1234"))
clf_model = CLFModel("./Chatty_intention/model_file")


def classifier(text):
    """
    判断是否是闲聊意图，以及是什么类型闲聊
    """
    return clf_model.predict(text)

def intent_classifier(text):
    url = "http://127.0.0.1:5000/service/api/bert_intent_recognize"
    data = {"text": text}
    headers = {'Content-Type':'application/json; charset=utf8'}
    reponse = requests.post(url, data=json.dumps(data), headers=headers)
    if reponse.status_code == 200:
        reponse = json.loads(reponse.text)
        return reponse["result"]
    else:
        return -1


def slot_recognizer(sample):
    url = 'http://127.0.0.1:5001/service/api/medical_ner'
    data = {"text": sample}
    headers = {'Content-Type': 'application/json;charset=utf8'}
    reponse = requests.post(url, data=json.dumps(data), headers=headers)
    if reponse.status_code == 200:
        reponse = json.loads(reponse.text)
        return reponse['result']
    else:
        return -1


# 如果属于闲聊意图, 直接返回准备好的模板
def gossip_robot(intent):
    return random.choice(gossip_corpus[intent])


def semantic_parser(text):
    intent_res = intent_classifier(text)
    slot_res = slot_recognizer(text)
    print("intent_res: ", intent_res)
    print("*"*100)
    print("slot_res: ", slot_res)
    print("*"*100)

    if intent_res == -1 or slot_res == -1 or len(slot_res) == 0 or intent_res.get("name") == "其他":
        return semantic_slot.get("unrecognized")

    slot_info = semantic_slot.get((intent_res["name"]))
    # print("slot_info: ", slot_info)

    # 填槽
    slots = slot_info.get("slot_list")
    slot_values = {}

    for key, value in slot_res.items():
        if value.lower() == slots[0].lower():
            slot_values[slots[0]] = key
    slot_info["slot_values"] = slot_values
    # print("new_slot_info: ", slot_info)

    # 根据意识强度来确定回复策略
    confi = intent_res.get("confidence")
    if confi >= intent_threshold_config["accept"]:
        slot_info["intent_strategy"] = "accept"
    elif confi >= intent_threshold_config["deny"]:
        slot_info["intent_strategy"] = "clarify"
    else:
        slot_info["intent_strategy"] = "deny"

    print("slot_info: ", slot_info)
    print("*"*100)
    return slot_info


# 根据语义槽获取答案数据
def get_answer(slot_info):
    cql_template = slot_info.get("cql_template")
    reply_template = slot_info.get("reply_template")
    ask_template = slot_info.get("ask_template")
    slot_values = slot_info.get("slot_values")
    strategy = slot_info.get("intent_strategy")

    if not slot_values:
        return slot_info

    if strategy == "accept":
        cql = []
        if isinstance(cql_template, list):
            for cqlt in cql_template:
                cql.append(cqlt.format(**slot_values))
        else:
            cql = cql_template.format(**slot_values)

        answer = neo4j_searcher(cql)
        print("answer: ", answer)
        print("*" * 100)
        if not answer:
            slot_info["replay_answer"] = "唔~我装满知识的大脑此刻很贫瘠"
        else:
            pattern = reply_template.format(**slot_values)
            slot_info["replay_answer"] = pattern + answer

    elif strategy == "clarify":
        # 澄清用户是否问该问题
        pattern = ask_template.format(**slot_values)
        slot_info["replay_answer"] = pattern
        # 得到肯定意图之后需要给用户回复的答案
        cql = []
        if isinstance(cql_template, list):
            for cqlt in cql_template:
                cql.append(cqlt.format(**slot_values))
        else:
            cql = cql_template.format(**slot_values)
        answer = neo4j_searcher(cql)
        print("answer: ", answer)
        if not answer:
            slot_info["replay_answer"] = "唔~我装满知识的大脑此刻很贫瘠"
        else:
            pattern = reply_template.format(**slot_values)
            slot_info["choice_answer"] = pattern + answer

    elif strategy == "deny":
        slot_info["replay_answer"] = slot_info.get("deny_response")
    print(slot_info)
    return slot_info




def neo4j_searcher(cql_list):
    result = ""
    if isinstance(cql_list, list):
        for cql in cql_list:
            # print(cql)
            res = []
            data = graph.run(cql).data()
            # print(data)
            if not data:
                continue
            for d in data:
                d = list(d.values())
                if isinstance(d[0], list):
                    res.extend(d[0])
                else:
                    res.extend(d)
            # print("res: ", res)
            data = "、".join([str(i) for i in res])
            # print("data: ", data)
            result += data + "\n"
        # print("result: ", result)
    else:
        data = graph.run(cql_list).data()
        if not data:
            return result
        res = []
        for d in data:
            d = list(d.values())
            if isinstance(d[0], list):
                res.extend(d[0])
            else:
                res.extend(d)
        data = "、".join([str(i) for i in res])
        result += data + "\n"
    # print("result: ", result)
    return result



# 如果确定诊断意图则使用该方法进行诊断问答
def medical_robot(text):
    slot_info = semantic_parser(text)
    answer = get_answer(slot_info)
    return answer


if __name__ == '__main__':
    # intent = classifier("你好啊")
    # print(gossip_rebot(intent))
    # print(slot_recognizer(sample="我朋友的父亲除了患有糖尿病，无手术外伤史及药物过敏史"))
    # print(intent_classifier(text="不同类型的肌无力症状表现有什么不同？"))
    # semantic_parser("你知道慢性支气管炎如何治疗吗?")
    medical_robot("高血压")