import torch
import json


class Config(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_path = r"C:\Users\19981\Documents\GitHub\MedicalKB\NER\LSTM_CRF\data\train.txt"
        self.vocab_path = r"C:\Users\19981\Documents\GitHub\MedicalKB\NER\LSTM_CRF\vocab\vocab.txt"
        self.embedding_dim = 300
        self.epochs = 10
        self.batch_size = 4
        self.hidden_dim = 256
        self.lr = 2e-3
        self.dropout = 0.2
        self.model = "BiLSTM_CRF"
        # self.model = "BiLSTM"
        self.tag2id = json.load(open(r"C:\Users\19981\Documents\GitHub\MedicalKB\NER\LSTM_CRF\data\tag2id.json", encoding="utf-8"))


if __name__ == "__main__":
    conf = Config()
    print(conf.train_path)
    print(conf.vocab_path)
    print(conf.tag2id)
