import torch
import torch.nn as nn
from TorchCRF import CRF
from utils.data_loader import *

class NERLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super(NERLSTM_CRF, self).__init__()
        self.name = "BiLSTM_CRF"
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id) + 1
        self.tag_to_ix = tag2id
        self.tag_size = len(tag2id)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

        #CRF
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.crf = CRF(self.tag_size)

    def forward(self, x, mask):
        #lstm模型得到的结果
        outputs = self.get_lstm2linear(x)
        outputs = outputs * mask.unsqueeze(-1)
        outputs = self.crf.viterbi_decode(outputs, mask)
        return outputs

    def log_likelihood(self, x, tags, mask):
        # lstm模型得到的结果
        outputs = self.get_lstm2linear(x)
        outputs = outputs * mask.unsqueeze(-1)
        # 计算损失
        return - self.crf(outputs, tags, mask)

    def get_lstm2linear(self, x):
        embedding = self.word_embeds(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        return outputs
if __name__ == '__main__':

    train_dataloader, dev_dataloader = get_data()
    embedding_dim = conf.embedding_dim
    hidden_dim = conf.hidden_dim
    dropout = conf.dropout
    tag2id = conf.tag2id
    my_model = NERLSTM_CRF(embedding_dim, hidden_dim, dropout, word2id, tag2id)
    for input_ids_padded, labels_padded, attention_mask in train_dataloader:
        print(input_ids_padded.shape)
        print(labels_padded.shape)
        print(attention_mask.shape)
        outputs = my_model(x=input_ids_padded, mask=attention_mask)
        print(f'outputs--》{outputs}')
        break
