import torch
import numpy as np
import argparse

from model import A2NN
from dataset import Traffic_Light
from torch.utils.data import DataLoader
from utils import get_train_val_names, check_folder


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset_path', type=str, default='TL_Dataset/')
    parse.add_argument('--remove_names', type=list, default=['README.txt',
                                                             'README.png',
                                                             'Testset'])
    parse.add_argument('--img_resize_shape', type=tuple, default=(64, 64))
    parse.add_argument('--num_workers', type=int, default=4)
    parse.add_argument('--val_size', type=float, default=0.3)
    parse.add_argument('--save_path', type=str, default='logs/')

    args = vars(parse.parse_args())

    check_folder(args['save_path'])

    # pylint: disable=E1101
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101

    model = A2NN().to(device)
    model.load_state_dict(torch.load(args['save_path']+'nn_state.t7'))

    model.eval()

    names = get_train_val_names(args['dataset_path'], args['remove_names'])

    val_dataset = Traffic_Light(names['val'], args['img_resize_shape'])

    val_dataload = DataLoader(val_dataset,
                              batch_size=1,
                              num_workers=args['num_workers'])

    count = 0
    for ind, (inp, label) in enumerate(val_dataload):
        inp = inp.float().to(device)
        label = label.long().to(device)
        output = model.forward(inp)
        output = np.argmax(output.to('cpu').detach().numpy(), axis=1)
        label = label.to('cpu').numpy()
        count += 1 if output == label else 0

    print('precision: {}'.format(count/(ind+1)))


if __name__ == "__main__":
    main()
