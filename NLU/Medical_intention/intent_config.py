import torch


class Config(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_path = r"C:\Users\19981\Documents\GitHub\MedicalKB\NLU\Medical_intention\data\train.csv"
        self.test_path = r"C:\Users\19981\Documents\GitHub\MedicalKB\NLU\Medical_intention\data\test.csv"
        self.label_path = r"C:\Users\19981\Documents\GitHub\MedicalKB\NLU\Medical_intention\data\label.txt"
        self.epochs = 10
        self.lr = 2e-5
        self.batch_size = 16
        self.max_len = 60
        self.num_class = 13
        self.bert_hiden = 768
        self.bert_path = r"C:\Users\19981\Documents\GitHub\MedicalKB\NLU\Medical_intention\save_model\bert-base-chinese"


if __name__ == '__main__':
    conf = Config()
