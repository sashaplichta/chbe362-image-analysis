import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class sigmoid():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def predict(self, x):
        return 1.0 / (1.0 + np.exp(self.a * x + self.b))

class preprocess():
    def __init__(self, image):
        self.image = cv2.imread(image)
        self.name = image[7:].split()[0]
        self.color = self.name.split("_")[0]
        self.conc = self.name.split("_")[1]
        self.t = self.name.split("_")[2]
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        x = np.arange(0, 1000).reshape(-1, 1)

        self.data = self.get_data(self.gray)
        # normalize data to between 0, 1
        self.data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))

        # Remove 1 and 0 (will cause over/underflow errors during scaling)
        self.data = np.where(self.data < 0.9999, self.data, 0.9999)
        self.data = np.where(self.data > 0.0001, self.data, 0.0001)

        # Scale data to linear

        i, j = self.get_linear(self.data)   
        print(i, j)     

        self.ydata = np.log((1 / self.data[i:j]) - 1)

        self.func = self.fit_func(x[i:j], self.ydata)

    def get_linear(self, x):
        for i in range(len(x)):
            if x[i] < 0.95:
                for j in range(len(x)):
                    if x[j] < 0.05:
                        return i, j

    def get_data(self, image):
        y, x = image.shape
        c_x = math.floor(x / 2)
        c_x2 = math.floor(x / 2 + 10)
        column_image = image[:, c_x:c_x2]
        data = self.horizontal_average(column_image)
        
        return data

    def horizontal_average(self, column_image):
        y, x = column_image.shape
        self.scale = x * 10
        column_average = np.zeros(y)
        for i in range(y):
            column_average[i] = np.average(column_image[i, :])
        
        column_average = 255 - column_average

        return np.flip(column_average)[:1000]
    
    def fit_func(self, x, y):
        model = LinearRegression().fit(x, y)
        alpha = model.coef_[0]
        beta = model.predict([[0]])[0]
        return sigmoid(alpha, beta)