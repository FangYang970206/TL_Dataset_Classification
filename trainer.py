import torch.nn as nn
from torch.optim import Adam


class Trainer:
    def __init__(self, model, dataload, epoch, lr, device):
        self.model = model
        self.dataload = dataload
        self.epoch = epoch
        self.lr = lr
        self.device = device
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def __epoch(self, epoch):
        self.model.train()
        loss_sum = 0
        for ind, (inp, label) in enumerate(self.dataload):
            inp = inp.float().to(self.device)
            label = label.long().to(self.device)
            self.optimizer.zero_grad()
            out = self.model.forward(inp)
            loss = self.criterion(out, label)
            loss.backward()
            loss_sum += loss.item()
            self.optimizer.step()
            print('epoch{}_step{}_train_loss_: {}'.format(epoch,
                                                          ind,
                                                          loss.item()))
        return loss_sum/(ind+1)

    def train(self):
        train_loss = self.__epoch(self.epoch)
        return train_loss
