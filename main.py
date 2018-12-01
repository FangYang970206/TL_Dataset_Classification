import torch
import argparse

from model import A2NN
from dataset import Traffic_Light
from utils import get_train_val_names, check_folder
from trainer import Trainer
from validator import Validator
from logger import Logger
from torch.utils.data import DataLoader


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset_path', type=str, default='TL_Dataset/')
    parse.add_argument('--remove_names', type=list, default=['README.txt',
                                                             'README.png',
                                                             'Testset'])
    parse.add_argument('--img_resize_shape', type=tuple, default=(32, 32))
    parse.add_argument('--batch_size', type=int, default=1024)
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--num_workers', type=int, default=4)
    parse.add_argument('--epochs', type=int, default=200)
    parse.add_argument('--val_size', type=float, default=0.3)
    parse.add_argument('--save_model', type=bool, default=True)
    parse.add_argument('--save_path', type=str, default='logs/')

    args = vars(parse.parse_args())

    check_folder(args['save_path'])

    # pylint: disable=E1101
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101

    model = A2NN().to(device)

    names = get_train_val_names(args['dataset_path'], args['remove_names'])

    train_dataset = Traffic_Light(names['train'], args['img_resize_shape'])
    val_dataset = Traffic_Light(names['val'], args['img_resize_shape'])

    train_dataload = DataLoader(train_dataset,
                                batch_size=args['batch_size'],
                                shuffle=True,
                                num_workers=args['num_workers'])

    val_dataload = DataLoader(val_dataset,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              num_workers=args['num_workers'])

    loss_logger = Logger(args['save_path'])

    logger_dict = {'train_losses': [],
                   'val_losses': []}

    for epoch in range(args['epochs']):
        print('<Main> epoch{}'.format(epoch))
        trainer = Trainer(model, train_dataload, epoch, args['lr'], device)
        train_loss = trainer.train()
        if args['save_model']:
            state = model.state_dict()
            torch.save(state, 'logs/nn_state.t7')
        validator = Validator(model, val_dataload, epoch,
                              device, args['batch_size'])
        val_loss = validator.eval()
        logger_dict['train_losses'].append(train_loss)
        logger_dict['val_losses'].append(val_loss['val_loss'])

        loss_logger.update(logger_dict)


if __name__ == '__main__':
    main()
