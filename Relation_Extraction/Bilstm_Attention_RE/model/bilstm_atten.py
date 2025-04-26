import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from data_loader import *
conf = Config()


class BiLSTM_Atten(nn.Module):
    def __init__(self, conf, vocab_size, pos_size, tag_size):
        super().__init__()
        self.device = conf.device
        self.embedding_dim = conf.embedding_dim
        self.pos_dim = conf.pos_dim
        self.hidden_dim = conf.hidden_dim
        self.batch_size = conf.batch_size
        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.tag_size = tag_size

        # 定义wordEmbedding层
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        # 定义相对实体1的位置编码
        self.p1Embed = nn.Embedding(self.pos_size, self.pos_dim)
        # 定义相对于实体2的位置编码
        self.p2Embed = nn.Embedding(self.pos_size, self.pos_dim)

        # 定义双向的LSTM
        self.lstm = nn.LSTM(input_size=self.embedding_dim+self.pos_dim*2, hidden_size=self.hidden_dim//2, bidirectional=True)
        # 定义输出层
        self.out = nn.Linear(in_features=self.hidden_dim, out_features=self.tag_size)

        # 定义三个dropout层
        self.embed_dropout = nn.Dropout(p=0.2)
        self.lstm_dropout = nn.Dropout(p=0.2)
        self.atten_dropout = nn.Dropout(p=0.2)

        # 定义一个注意力权重参数
        self.atten_weight = nn.Parameter(torch.randn(self.batch_size, 1, self.hidden_dim).to(conf.device))

    def init_lstm_hidden(self):
        h0 = torch.zeros(2, self.batch_size, self.hidden_dim//2).to(conf.device)
        c0 = torch.zeros(2, self.batch_size, self.hidden_dim//2).to(conf.device)
        return h0, c0

    def attention(self, H):
        # H [4, 200, 70]
        M = F.tanh(H)
        a = F.softmax(torch.bmm(self.atten_weight, M), dim=-1)  # [4, 1, 200], [4, 200, 70] -> [4, 1, 70]
        a = torch.transpose(a, 1, 2)  # [4, 70, 1]
        return torch.bmm(H, a)  # [4, 200, 70], [4, 70, 1]  -> [4, 200, 1]

    def forward(self, sentence, pos1, pos2):
        # [4, 70], [4, 70], [4, 70] -> [4, 70, 128], [4, 70, 32], [4, 70, 32]
        embeds = torch.cat((self.embed(sentence), self.p1Embed(pos1), self.p2Embed(pos2)), dim=-1)  # [4, 70, 192]
        embeds = self.embed_dropout(embeds)
        embeds = torch.transpose(embeds, 0, 1)  # [70, 4, 192]

        # 初始化: h0和c0
        init_hidden = self.init_lstm_hidden()
        lstm_out, lstm_hiddem = self.lstm(embeds, init_hidden)  # [70, 4, 200]
        lstm_out = lstm_out.permute(1, 2, 0)  # [4, 200, 70]
        lstm_out = self.lstm_dropout(lstm_out)

        atten_out = F.tanh(self.attention(lstm_out))  # [4, 200, 1]
        atten_out = self.atten_dropout(atten_out).squeeze()  # [4, 200]

        result = self.out(atten_out)  # [4, 5]
        return result


def test_model():
    vocab_size = len(word2id)
    pos_size = 143
    tag_size = len(relation2id)
    ba = BiLSTM_Atten(conf, vocab_size, pos_size, tag_size)
    ba.to(conf.device)
    train_dataloader, test_dataloader = get_data_loader()
    for inputs, positionE1, positionE2, labels, _, _, _, in train_dataloader:
        out = ba(inputs, positionE1, positionE2)
        print(out.shape)
        break



if __name__ == "__main__":
    test_model()



