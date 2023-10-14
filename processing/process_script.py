import os
from preprocess import preprocess
from scipy.special import erf, erfinv
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ficks_regression import match

class ficks():
    def __init__(self, c_s, c_0, D):
        self.c_s = c_s
        self.c_0 = c_0
        self.D = D

    def predict(self, X, t):
        return self.c_s - (self.c_s - self.c_0) * erf(X / (2.0 * np.sqrt(self.D * t)))

def run_script(verbose=False):
    shortcut = {"0" : 0, "1" : 1, "20" : 1, "40" : 2, "60" : 3}

    image_names = os.listdir("images/")
    processed_images = []
    data = []

    for n in image_names:
        if n[-4:] == "jpeg":
            proc_n = preprocess("images/" + n)
            processed_images.append(proc_n)
            data.append(proc_n.data)

    # c_s = 0.5
    # c_0 = 0.0
    # Ds = {'red' : {},
    #       'green' : {}}
    # for i in processed_images:
    #     if i.t == 0: 
    #         t = 0.25
    #     else:
    #         t = 2. * 7. * 24.
    #     x = np.arange(-200, 200).reshape(-1, 1)
    #     func = i.shifted_func
    #     trans_data = 2 * erfinv((func.predict(x) - c_s) / (-1 * (c_s - c_0))) # gives 1/sqrt(D*t) * X
    #     model = LinearRegression().fit(x, trans_data)
    #     alpha = model.coef_[0]

    #     i.D = 1. / alpha**2 / t
    #     i.fick = ficks(c_s, c_0, i.D)
    #     Ds[i.color][i.conc + "_" + i.t] = i.D

    # red_df = pd.DataFrame(Ds['red'])
    # green_df = pd.DataFrame(Ds['green'])
    # print(red_df)
    # print(green_df)


    if verbose:
        fig, axs = plt.subplots(2, 4)
        fig.suptitle("Red + Green")
        for i in processed_images:
            x = np.arange(0, 1000)
            if i.t == 0: 
                t = 0.25
            else:
                t = 2. * 7. * 24.

            ax = axs[shortcut[i.t]][shortcut[i.conc]]
            ax.plot(x, i.data, label=i.name, color=i.color)
            ax.plot(x, i.func.predict(x), label="Ideal Sigmoid", color="blue")
            # ax.plot(x, i.fick.predict(x, t), label="Ficks Solution", color="Orange")
            ax.set_title(i.color + "_" + i.conc + "_" + i.t)
            ax.legend()

        plt.show()

run_script(verbose=True)