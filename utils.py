import os
from glob import glob


def get_train_val_names(dataset_path, remove_names, radio=0.3):
    train_names = []
    val_names = []
    dataset_paths = os.listdir(dataset_path)
    for n in remove_names:
        dataset_paths.remove(n)
    for path in dataset_paths:
        sub_dataset_path = os.path.join(dataset_path, path)
        sub_dataset_names = glob(os.path.join(sub_dataset_path, '*.png'))
        sub_dataset_len = len(sub_dataset_names)
        val_names.extend(sub_dataset_names[:int(radio*sub_dataset_len)])
        train_names.extend(sub_dataset_names[int(radio*sub_dataset_len):])
    return {'train': train_names, 'val': val_names}


def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
