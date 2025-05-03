import time
from slot_predict import *
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/service/api/medical_ner", methods=["GET","POST"])
def medical_ner():
    data = {"sucess":0}
    result = None
    param = request.get_json()
    text = param["text"]
    try:
        result = model2test(text)
        data["result"] = result
        data["sucess"] = 1
    except:
        print('模型调用有误')

    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
