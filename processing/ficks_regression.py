import os
from preprocess import preprocess
from preprocess import sigmoid
from scipy.special import erf, erfinv
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ficks():
    def __init__(self, c_s, c_0, D):
        self.c_s = c_s
        self.c_0 = c_0
        self.D = D

    def predict(self, X, t):
        return self.c_s - (self.c_s - self.c_0) * erf(X / (2.0 * np.sqrt(self.D * t)))
    
def match(s, t):
    c_s = 0.5
    c_0 = 0.0
    shifted_s = sigmoid(s.a, 0)
    x = np.arange(-100, 100).reshape(-1, 1)
    y = shifted_s.predict(x)
    trans_y = 2 * erfinv((y - c_s) / (-1 * (c_s - c_0)))
    trans_y = np.where(trans_y > 0.00001, trans_y, 0.00001)
    trans_y = np.where(trans_y < 100, trans_y, 100)

    for i in range(len(x)):
        print(x[i], trans_y[i])

    model = LinearRegression().fit(x, trans_y)
    alpha = model.coef_[0]
    D = 1. / alpha**2 / t

    f = ficks(c_s, c_0, D)

    return f

# test_s = sigmoid(1, 0)
# test_f = match(test_s, 1)

# x = np.arange(-10 , 10)

# plt.plot(x, test_s.predict(x), label="sigmoid", color="blue")
# plt.plot(x, test_f.predict(x, 1), label="fick", color="red")
# plt.show()