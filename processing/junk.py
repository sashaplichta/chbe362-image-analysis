import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, erfinv
from sklearn.linear_model import LinearRegression
c_s = 0.5
c_0 = 0
b = 0
D = 10
t = 1

xs = np.arange(-500, 500)

def s(X): return 1 / (1 + np.exp(0.04795140368419283 * X))
def f(X): return c_s - (c_s - c_0) * erf((X + b) / (2.0 * np.sqrt(D * t)))
def fi(y): return (erfinv((y - c_s) / (c_s - c_0)) * 2.0) * np.sqrt(t)

ys = s(xs)

ys = fi(ys)
model = LinearRegression(fit_intercept=False)
model.fit(xs.reshape(-1, 1), ys.reshape(-1, 1))
D = (1 / model.coef_[0][0]) ** 2
print(D)

plt.plot(xs, s(xs))
plt.plot(xs, f(xs))

plt.show()