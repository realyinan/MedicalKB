import os
import json


LOGS_DIR = r"C:\Users\19981\Documents\GitHub\MedicalKB\NLU\logs"


def dump_user_dialogue_context(data):
    path = os.path.join(LOGS_DIR, "{}.json".format("user"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, indent=4, separators=(", ", ": "), ensure_ascii=False)


def load_user_dialogue_context():
    path = os.path.join(LOGS_DIR, "{}.json".format("user"))
    if not os.path.exists(path):
        return {"choice_answer": "hi，机器人小智很高心为您服务", "slot_values": None}
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
            return json.loads(data)


if __name__ == '__main__':
    data = {'slot_list': ['Disease'],
            'slot_values': {'Disease': '慢性支气管炎'},
            'cql_template': "MATCH(p:疾病) WHERE p.name='{DiRETURN p.desc",
            'reply_template': "'{Disease}' 是这样的：\n",
            'ask_template': "您问的是 '{Disease}' 的定义吗？"}
    # dump_user_dialogue_context(data)
    print(load_user_dialogue_context())