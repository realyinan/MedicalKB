from torch.utils.data import Dataset, DataLoader
from process import *
import torch


word2id, id2word = get_word_id(conf.train_data_path)
class MyDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = get_text_data(data_path)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, item):
        sequence = self.data[0][item]
        label = int(self.data[1][item])
        position1 = self.data[2][item]
        position2 = self.data[3][item]
        entity = self.data[4][item]
        return sequence, label, position1, position2, entity


def collate_fn(values):
    sequences = [value[0] for value in values]
    labels = [value[1] for value in values]
    position1s = [value[2] for value in values]
    position2s = [value[3] for value in values]
    entities= [value[4] for value in values]

    sequences_ids = []
    for words in sequences:
        ids = sent_padding(words, word2id)
        sequences_ids.append(ids)

    positionsE1_ids = []
    positionsE2_ids = []
    for pos1_ids in position1s:
        pos_ids = position_padding(pos1_ids)
        positionsE1_ids.append(pos_ids)
    for pos2_ids in position2s:
        pos_ids = position_padding(pos2_ids)
        positionsE2_ids.append(pos_ids)

    input_tensor = torch.tensor(sequences_ids, dtype=torch.long, device=conf.device)
    positionsE1_tensor = torch.tensor(positionsE1_ids, dtype=torch.long, device=conf.device)
    positionsE2_tensor = torch.tensor(positionsE2_ids, dtype=torch.long, device=conf.device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=conf.device)
    return input_tensor, positionsE1_tensor, positionsE2_tensor, labels_tensor, sequences, labels, entities


def get_data_loader():
    train_dataset = MyDataset(conf.train_data_path)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=conf.batch_size, drop_last=True, collate_fn=collate_fn)

    test_dataset = MyDataset(conf.test_data_path)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=conf.batch_size, drop_last=True, collate_fn=collate_fn)
    return train_dataloader, test_dataloader


def test_data():
    dt = MyDataset(conf.train_data_path)
    sequence, label, position1, position2, entity = dt[0]
    print(sequence)
    print(label)
    print(position1)
    print(position2)
    print(entity)


if __name__ == "__main__":
    # test_data()
    train_dataloader, test_dataloader = get_data_loader()
    for inputs, positionE1, positionE2, labels, _, _, _, in test_dataloader:
        print(inputs)
        print(positionE1)
        print(positionE2)
        print(labels)
        break
