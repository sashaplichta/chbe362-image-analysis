import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class preprocess():
    def __init__(self, image):
        self.image = cv2.imread(image)
        self.name = image[7:].split()[0]
        self.color = self.name.split("_")[0]
        self.conc = self.name.split("_")[1]
        self.t = self.name.split("_")[2]
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.data = self.get_data(self.gray)

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
    
    def plot(self):
        plt.plot(np.arange(0, len(self.data)), self.data, label="green", color="green")
        plt.legend()
        plt.show()

test = preprocess("images/green_40_0 - IMG_3205.jpeg")
test.plot()