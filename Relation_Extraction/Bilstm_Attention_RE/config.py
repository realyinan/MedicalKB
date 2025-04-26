import torch


class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data_path = r"C:\Users\19981\Documents\GitHub\MedicalKB\Relation_Extraction\Bilstm_Attention_RE\data\train.txt"
        self.test_data_path = r"C:\Users\19981\Documents\GitHub\MedicalKB\Relation_Extraction\Bilstm_Attention_RE\data\test.txt"
        self.rel_data_path = r"C:\Users\19981\Documents\GitHub\MedicalKB\Relation_Extraction\Bilstm_Attention_RE\data\relation2id.txt"
        self.embedding_dim = 128
        self.pos_dim = 32
        self.hidden_dim = 200
        self.epochs = 50
        self.batch_size = 32
        self.max_len = 70
        self.learning_rate = 1e-3


if __name__ == "__main__":
    conf = Config()
    print(conf.train_data_path)