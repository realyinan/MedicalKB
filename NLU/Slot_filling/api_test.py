import json
import requests


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

if __name__ == '__main__':
    result = slot_recognizer(sample="我朋友的父亲除了患有糖尿病，无手术外伤史及药物过敏史")
    print(result)