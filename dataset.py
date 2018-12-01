import torch
import cv2
import torch.utils.data as data


class_light = {
                'Red Circle': 0,
                'Green Circle': 1,
                'Red Left': 2,
                'Green Left': 3,
                'Red Up': 4,
                'Green Up': 5,
                'Red Right': 6,
                'Green Right': 7,
                'Red Negative': 8,
                'Green Negative': 8
}


class Traffic_Light(data.Dataset):
    def __init__(self, dataset_names, img_resize_shape):
        super(Traffic_Light, self).__init__()
        self.dataset_names = dataset_names
        self.img_resize_shape = img_resize_shape

    def __getitem__(self, ind):
        img = cv2.imread(self.dataset_names[ind])
        img = cv2.resize(img, self.img_resize_shape)
        img = img.transpose(2, 0, 1)-127.5/127.5
        for key in class_light.keys():
            if key in self.dataset_names[ind]:
                label = class_light[key]
        # pylint: disable=E1101,E1102
        return torch.from_numpy(img), torch.tensor(label)
        # pylint: disable=E1101,E1102

    def __len__(self):
        return len(self.dataset_names)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from glob import glob
    import os

    path = 'TL_Dataset/Green Up/'
    names = glob(os.path.join(path, '*.png'))
    dataset = Traffic_Light(names, (64, 64))
    dataload = DataLoader(dataset, batch_size=1)
    for ind, (inp, label) in enumerate(dataload):
        print("{}-inp_size:{}-label_size:{}".format(ind, inp.numpy().shape,
                                                    label.numpy().shape))
