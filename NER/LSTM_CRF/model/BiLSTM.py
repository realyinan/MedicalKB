import torch.nn as nn
from data_loader import *


class NERBiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super().__init__()
        self.name = "BiLSTM"
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id) + 1
        self.tag_len = len(tag2id)
        self.tag2id = tag2id

        self.dropout = nn.Dropout(p=dropout)
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.out = nn.Linear(self.hidden_dim, self.tag_len)

    def forward(self, x, mask):
        # x -> [batch_size, seq_len]
        # mask -> [batch_size, seq_len]
        embed_x = self.embed(x)  # [batch_size, seq_len, embedding_dim]
        output, hidden = self.lstm(embed_x)  # [batch_size,seq_len, hidden_dim]
        # 只保留有效的输出结果
        output = output * mask.unsqueeze(-1)
        output = self.dropout(output)
        result = self.out(output)  # [batch_size, seq_len, tag_len]
        return result


def test_model():
    conf = Config()
    embedding_dim = conf.embedding_dim
    hidden_dim = conf.hidden_dim
    dropout = conf.dropout
    tag2id = conf.tag2id
    lstm = NERBiLSTM(embedding_dim, hidden_dim, dropout, word2id, tag2id)
    print(lstm)

    train_dataloader, dev_dataloader = get_data()
    for inputs, labels, attention_mask in train_dataloader:
        output = lstm(inputs, attention_mask)
        print(output.shape)
        break

if __name__ == "__main__":
    test_model()



