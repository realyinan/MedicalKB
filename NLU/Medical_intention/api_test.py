import requests
import json


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


if __name__ == '__main__':
    result = intent_classifier(text="不同类型的肌无力症状表现有什么不同？")
    print(result)