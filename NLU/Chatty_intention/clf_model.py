import os
import pickle
import numpy as np


class CLFModel(object):
    def __init__(self, model_data_path):
        super().__init__()
        self.id2label = pickle.load(open(os.path.join(model_data_path, "id2label.pkl"), "rb"))
        self.vec = pickle.load(open(os.path.join(model_data_path, "vec.pkl"), "rb"))
        self.LR = pickle.load(open(os.path.join(model_data_path, "LR.pkl"), "rb"))
        self.gbdt = pickle.load(open(os.path.join(model_data_path, "gbdt.pkl"), "rb"))

    def predict(self, text):
        text = " ".join(list(text.lower()))
        x = self.vec.transform([text])
        prab1 = self.LR.predict_proba(x)
        prab2 = self.gbdt.predict_proba(x)
        label = np.argmax((prab1+prab2)/2, axis=1)
        predict = self.id2label[label[0]]
        return predict




if __name__ == '__main__':
    model = CLFModel("./model_file")
    text = "你好吗"
    result = model.predict(text)
    print(result)
