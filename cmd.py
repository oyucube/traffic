from dataset.base import MyDataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

path = "C:/Users/waka-lab/Documents/data/data/mydata/"
dataset = MyDataset(path)
image, label = dataset.get_example(10)

c = np.arange(24).reshape(4, 3, 2)
print(c.shape)
print(type(c))
c = c.transpose(1, 2, 0)
print(c.shape)

print(image.shape)
print(type(image))
t_image = image.transpose(1, 2, 0)
print(t_image.shape)

plt.imshow(t_image)
plt.show()

