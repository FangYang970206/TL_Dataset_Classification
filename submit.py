import torch
import numpy as np
import argparse
import os
import cv2

from model import A2NN
from utils import check_folder


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset_path', type=str,
                       default='TL_Dataset/Testset/')
    parse.add_argument('--img_resize_shape', type=tuple, default=(32, 32))
    parse.add_argument('--num_workers', type=int, default=4)
    parse.add_argument('--save_path', type=str, default='logs/')

    args = vars(parse.parse_args())

    check_folder(args['save_path'])

    # pylint: disable=E1101
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pylint: disable=E1101

    model = A2NN().to(device)
    model.load_state_dict(torch.load(args['save_path']+'nn_state.t7'))

    model.eval()

    txt_path = os.path.join(args['save_path'], 'result.txt')
    with open(txt_path, 'w') as f:
        for i in range(20000):
            name = os.path.join(args['dataset_path'], '{}.png'.format(i))
            img = cv2.imread(name)
            img = cv2.resize(img, args['img_resize_shape'])
            img = img.transpose(2, 0, 1)-127.5/127.5
            img = torch.unsqueeze(torch.from_numpy(img).float(), dim=0)
            img = img.to(device)
            output = model.forward(img).to('cpu').detach().numpy()
            img_class = np.argmax(output, axis=1)
            f.write(name.split('/')[2] + ' ' + str(img_class[0]))
            f.write('\n')


if __name__ == "__main__":
    main()
