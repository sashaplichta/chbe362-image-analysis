import os
from preprocess import preprocess
from preprocess import sigmoid
from scipy.special import erf, erfinv
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ficks():
    def __init__(self, c_s, c_0, D, b):
        self.c_s = c_s
        self.c_0 = c_0
        self.D = D
        self.b = b

    def predict(self, X, t):
        return self.c_s - (self.c_s - self.c_0) * erf((X + self.b) / (2.0 * np.sqrt(self.D * t)))
    
    def inverse(self, y, t): 
        return (erfinv((y - self.c_s) / (self.c_s - self.c_0)) * 2.0) * np.sqrt(t)
    
def match(s, t):
    colors = ["orange", "purple", "pink", "olive"]
    c_s = 0.5
    c_0 = 0.0
    shifted_s = sigmoid(s.a, 0)
    # print(s.a)
    # f = ficks(c_s, c_0, None, 0)

    # xs = np.arange(-100, 100)
    # ys = shifted_s.predict(xs)
    # ys = f.inverse(ys, 1)

    # model = LinearRegression(fit_intercept=False)
    # model.fit(xs.reshape(-1, 1), ys.reshape(-1, 1))
    # D = (1 / model.coef_[0][0]) ** 2


    # f = ficks(c_s, c_0, D, s.b / s.a)
    D = 1 / (5 * s.a)
    # f = ficks(c_s, c_0, D, 0)

    x = np.arange(-500, 500, 1)
    inp = None
    while inp != "end":
        f = ficks(c_s, c_0, D, 0)
        plt.plot(x, shifted_s.predict(x), label="sigmoid", color="blue")
        plt.plot(x, f.predict(x, t), label="fick", color="red")
        plt.show()
        print("Prev: ", D)
        inp = input()
        if inp != "end": D = float(inp)
    print("ended loop")
    b = (s.b / s.a)
    f = ficks(c_s, c_0, D, b)
    return f

# def match(s, t):
#     colors = ["orange", "purple", "pink", "olive"]

#     c_s = 0.5
#     c_0 = 0.0
#     shifted_s = sigmoid(s.a, 0)
#     epsilon = 0.001
#     D = 1. / (5. * s.a)
#     f = ficks(c_s, c_0, D, 0)
#     delta = f.predict(0.01, t) - s.predict(0.01)
#     it = 0
#     limit = 20000
#     print(np.log(3) / s.a)
#     while abs(delta) > epsilon and it < limit:
#         D -= delta * 0.01
#         f = ficks(c_s, c_0, D, 0)
#         delta = f.predict(np.log(9) / s.a, t) - s.predict(np.log(9) / s.a)
#         if it % 5000 == 0:
#             x = np.arange(-1000, 1000, 1)
#             plt.plot(x, f.predict(x, t), label="fick", color=colors[int(it / 5000)])
#         it += 1
#     print(it, D)
#     # f = ficks(c_s, c_0, D, s.b / s.a)
#     f = ficks(c_s, c_0, D, 0)

#     x = np.arange(-1000, 1000, 1)

#     plt.plot(x, shifted_s.predict(x), label="sigmoid", color="blue")
#     plt.plot(x, f.predict(x, t), label="fick", color="red")
#     plt.show()

#     return f

# test_s = sigmoid(10, 0)
# # test_f = ficks(.5, 0, 0.01, 0)
# test_f = match(test_s, 1)

# x = np.arange(-1, 1, 0.1)

# plt.plot(x, test_s.predict(x), label="sigmoid", color="blue")
# plt.plot(x, test_f.predict(x, 1), label="fick", color="red")
# plt.show()