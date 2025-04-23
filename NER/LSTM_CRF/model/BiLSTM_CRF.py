import torch.nn as nn
from TorchCRF import CRF
from data_loader import *
import torch



class NERBiLSTMCRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super().__init__()
        self.name = "BiLSTM_CRF"
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id) + 1
        self.tag_len = len(tag2id)
        self.tag2id = tag2id

        self.dropout = nn.Dropout(p=dropout)
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.out = nn.Linear(self.hidden_dim, self.tag_len)
        # 定义CRF层
        self.crf = CRF(self.tag_len)

    def forward(self, x, mask):
        outputs = self.get_lstm2linear(x)
        outputs = outputs * mask.unsqueeze(-1)
        outputs = self.crf.viterbi_decode(outputs, mask)
        return outputs

    def log_likelihood(self, x, tags, mask):
        outputs = self.get_lstm2linear(x)
        outputs = outputs * mask.unsqueeze(-1)
        # 计算损失
        return - self.crf(outputs, tags, mask)

    def get_lstm2linear(self, x):
        embedding = self.embed(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.out(outputs)
        return outputs


def test_model():
    conf = Config()
    embedding_dim = conf.embedding_dim
    hidden_dim = conf.hidden_dim
    dropout = conf.dropout
    tag2id = conf.tag2id
    lstm = NERBiLSTMCRF(embedding_dim, hidden_dim, dropout, word2id, tag2id)
    print(lstm)
    train_dataloader, dev_dataloader = get_data()
    for inputs, labels, attention_mask in train_dataloader:
        attention_mask = attention_mask.to(torch.bool)
        result = lstm.log_likelihood(inputs, labels, attention_mask)
        print(result)
        break


if __name__ == "__main__":
    test_model()

