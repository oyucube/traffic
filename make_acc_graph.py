import matplotlib.pyplot as plt
import numpy as np

# acc_list = ["normal_train", "normal_test", "pretraintrain", "pretraintest", "pretrain_fix_train", "pretrain_fix_test"
#     , "2_step_train", "2_step_test", "VGG_train", "VGG_test"]
acc_list = ["p24train", "p24test", "p24v2train", "p24v2test"]
line_color = ["blue", "blue", "green", "green"]

# line_color = ["red", "red", "blue", "blue", "green", "green", "purple", "purple", "gold", "gold"]

plt.figure()
# plt.rcParams["font.size"] = 18
plt.xlim([0, 50])
plt.ylim([0, 1])
for item, lc in zip(acc_list, line_color):
    acc = np.load("graph/" + item + ".npy")
    if item.find("test") != -1:
        plt.plot(acc, label=item, color=lc, marker=".", alpha=0.5)
    else:
        plt.plot(acc, label=item, color=lc)
plt.xlabel("epoch")
plt.ylabel("accuracy")
# plt.legend(loc='upper right', bbox_to_anchor=(1.05, 0.5, 0.5, .100), borderaxespad=0., )
# plt.legend(loc="lower right")
# plt.subplots_adjust(right=0.7)
plt.legend()
plt.savefig("graph/graph.png")
