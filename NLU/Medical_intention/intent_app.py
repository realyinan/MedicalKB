import time
import torch
import torch.optim as optim
from intent_config import *
from flask import Flask, request, jsonify
from model import *
app = Flask(__name__)
conf = Config()


# 获得所有标签类别
label_list = [line.strip() for line in  open(conf.label_path, "r", encoding="utf-8")]
id2label = {idx: label for idx, label in enumerate(label_list)}

bert_tokenizer = BertTokenizer.from_pretrained(conf.bert_path)
bert_model = MyModel(conf.bert_path, conf.bert_hiden, conf.num_class)
bert_model.load_state_dict(torch.load("./save_model/epoch_10.pth"))
bert_model.to(conf.device)

def modle2predict(sample, model):
    inputs = bert_tokenizer.encode_plus(sample, padding="max_length", truncation=True, max_length=60, return_tensors="pt")
    input_ids = inputs["input_ids"].to(conf.device)
    attention_mask = inputs["attention_mask"].to(conf.device)
    token_type_ids = inputs["token_type_ids"].to(conf.device)

    model.eval()
    with torch.no_grad():
        logists = model(input_ids, attention_mask, token_type_ids)
        logists = torch.softmax(logists, dim=-1)
        out = torch.argmax(logists, dim=-1).item()  # [batch_size]
        v, k = torch.topk(logists, k=1)  # [batch_size, k]
        return {"name": id2label[out], "confidence": round(float(v.item()), 3)}


@app.route("/service/api/bert_intent_recognize", methods=["GET", "POST"])
def service():
    data = {"sucess": 0}
    result = None
    param = request.get_json()
    text = param["text"]

    try:
        result = modle2predict(text, bert_model)
        data["result"] = result
        data["sucess"] = 1
    except:
        print("模型调用有误")
    return jsonify(data)



if __name__ == '__main__':
    # result = modle2predict("不同类型的肌无力症状表现有什么不同？", bert_model)
    # print(result)
    app.run(host="0.0.0.0", port=5000)
