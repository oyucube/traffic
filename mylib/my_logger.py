import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class LOGGER:
    def __init__(self, log_file_path, file_id, log_mode=0, n_epoch=30):
        self.log_file_path = log_file_path
        self.file_id = file_id
        self.log_mode = log_mode
        self.log_buf = []
        self.val_array = np.full_like(np.zeros(n_epoch), np.nan)
        self.train_array = np.full_like(np.zeros(n_epoch), np.nan)
        self.loss_array = np.full_like(np.zeros(n_epoch), np.nan)
        self.baseline = np.full_like(np.zeros(n_epoch), np.nan)
        self.max_acc = 0
        self.best = ""

        with open(log_file_path + "log.txt", "w") as f:
            f.write(" ")

    def l_print(self, sentence):
        print(sentence)
        self.log_buf.append(sentence)
        return

    def update_log(self):
        with open(self.log_file_path + "log.txt", "a") as f:
            for buf in self.log_buf:
                f.write(buf)
        self.log_buf = []
        return

    def set_loss(self, loss, epoch):
        self.loss_array[epoch] = loss

    def set_acc(self, train, val, epoch):
        self.train_array[epoch] = train
        self.val_array[epoch] = val
        self.best = ""
        if val > self.max_acc:
            self.max_acc = val
            self.best = "best_"
        self.l_print("train_acc:{:1.4f} test_acc:{:1.4f}\n".format(train, val))

    def set_baseline(self, b, epoch):
        self.baseline[epoch] = b

    def save_acc(self):
        np.save(self.log_file_path + self.file_id + "test.npy", self.val_array)
        np.save(self.log_file_path + self.file_id + "train.npy", self.train_array)
        np.save(self.log_file_path + self.file_id + "base.npy", self.baseline)
        plt.figure()
        plt.ylim([0, 1])
        p1 = plt.plot(self.val_array, color="green")
        p2 = plt.plot(self.train_array, color="blue")
        plt.legend((p1[0], p2[0]), ("test", "train"), loc=2)
        plt.savefig(self.log_file_path + self.file_id + "acc.png")
        plt.figure()
        plt.plot(self.loss_array)
        plt.savefig(self.log_file_path + self.file_id + "loss.png")
        plt.figure()
        plt.plot(self.baseline)
        plt.savefig(self.log_file_path + self.file_id + "base.png")
        plt.close("all")
