from collections import defaultdict
import numpy as np
import pdb
import matplotlib.pyplot as plt


class Recorder():
    def __init__(self, name):
        self.record = defaultdict(list)
        self.name = name

    def update(self, key, data, epoch):
        self.record[key].append([epoch, data])

    def save(self):
        np.save('../fig/' + self.name + '.pkl', self.record)

class Drawer():
    def __init__(self, path, image_path):
        self.record = np.load(path).tolist()
        self.image_path = image_path

    def plot(self, key, type, label):
        support = np.array(self.record[key])
        support_x = support[:, 0]
        support_y = support[:, 1]
        plt.plot(support_x, support_y, type, label=label)

    def draw(self):
        keys = list(self.record.keys())
        # plot loss
        loss_keys = [key for key in keys if 'loss' in key]
        for key in loss_keys:
            self.plot(key, '-', key)
        plt.title('loss vs. epoches')
        plt.ylabel('train loss')
        plt.legend(loc='best')
        # plt.show()
        plt.savefig(self.image_path)

        # plot acc
        acc_keys = [key for key in keys if 'acc' in key]
        for key in acc_keys:
            self.plot(key, '-', key)
        plt.title('acc vs. epoches')
        plt.ylabel('acc')
        plt.legend(loc='best')
        # plt.show()
        plt.savefig(self.image_path)

        # plot mAP:
        mAP_keys = [key for key in keys if 'mAP' in key]
        for key in mAP_keys:
            self.plot(key, '-', key)
        plt.title('mAP vs. epoches')
        plt.ylabel('mAP')
        plt.legend(loc='best')
        # plt.show()
        plt.savefig(self.image_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Semi-POINT')
    # Training settings
    parser.add_argument('-p', type=str, default='')
    args = parser.parse_args()

    drawer = Drawer(args.p, '../fig/tmp.png')
    drawer.draw()



