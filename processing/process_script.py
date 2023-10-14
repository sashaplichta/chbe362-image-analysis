import os
from preprocess import preprocess
import numpy as np
import matplotlib.pyplot as plt

shortcut = {"0" : 0, "1" : 1, "20" : 1, "40" : 2, "60" : 3}

image_names = os.listdir("images/")
processed_images = []
data = []

for n in image_names:
    if n[-4:] == "jpeg":
        print(n)
        proc_n = preprocess("images/" + n)
        processed_images.append(proc_n)
        data.append(proc_n.data)

fig, axs = plt.subplots(2, 4)
fig.suptitle("Red + Green")
for i in processed_images:
    print(i.name)
    ax = axs[shortcut[i.t]][shortcut[i.conc]]
    ax.plot(np.arange(0, 1000), i.data, label=i.name, color=i.color)
    ax.plot(np.arange(0, 1000), i.func.predict(np.arange(0, 1000).reshape(-1, 1)), label="Ideal", color="blue")
    ax.set_title(i.color + "_" + i.conc + "_" + i.t)

plt.show()
