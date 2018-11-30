import matplotlib.pyplot as plt
import os


class Logger:
    def __init__(self, save_path):
        self.save_path = save_path

    def update(self, Kwarg):
        self.__plot(Kwarg)

    def __plot(self, Kwarg):
        save_img_path = os.path.join(self.save_path, 'learning_curve.png')
        plt.clf()
        plt.plot(Kwarg['train_losses'], label='Train', color='g')
        plt.plot(Kwarg['val_losses'], label='Val', color='b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title('learning_curve')
        plt.savefig(save_img_path)
