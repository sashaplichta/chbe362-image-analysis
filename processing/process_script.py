import os
from preprocess import preprocess
import numpy as np
import matplotlib.pyplot as plt
from ficks_regression import ficks

shortcut = {"0" : 0, "1" : 1, "20" : 1, "40" : 2, "60" : 3}
times = {"0" : 0.25, "1" : 2*7*24}

image_names = os.listdir("images/")
processed_images = []
data = []

for n in image_names:
    if n[-4:] == "jpeg":
        proc_n = preprocess("images/" + n)
        processed_images.append(proc_n)
        data.append(proc_n.data)

ds = {'red_20_0': 4300.0, 'red_0_1': 175.0, 'red_20_1': 330.0, 'red_0_0': 2800.0,
      'green_40_1': 55.0, 'green_40_0': 1300.0, 'green_60_1': 2.0, 'green_60_0': 2200.0}
for i in processed_images:
    t = times[i.t]
    i.fick = ficks(0.5, 0, ds[i.name], i.func.b / i.func.a)


fig, axs = plt.subplots(2, 4)
fig.suptitle("Red + Green")
for i in processed_images:
    x = np.arange(0, 1000)
    t = times[i.t]
    ax = axs[shortcut[i.t]][shortcut[i.conc]]
    ax.plot(x, i.data, label=i.name, color=i.color)
    ax.plot(x, i.func.predict(x), label="sigmoid", color="blue")
    ax.plot(x, i.fick.predict(x, t), label="fick", color="orange")
    ax.legend()
    ax.set_title(i.color + "_" + i.conc + "_" + i.t)

plt.show()