import torch.nn as nn


class Validator:
    def __init__(self, model, dataload, epoch, device, batch_size):
        self.model = model
        self.dataload = dataload
        self.epoch = epoch
        self.device = device
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def __epoch(self, epoch):
        self.model.eval()
        loss_sum = 0
        for ind, (inp, label) in enumerate(self.dataload):
            inp = inp.float().to(self.device)
            label = label.long().to(self.device)
            out = self.model.forward(inp)
            loss = self.criterion(out, label)
            loss_sum += loss.item()
        return {'val_loss': loss_sum/(ind+1)}

    def eval(self):
        val_loss = self.__epoch(self.epoch)
        return val_loss
