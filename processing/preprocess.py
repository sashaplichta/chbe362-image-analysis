import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class preprocess():
    def __init__(self, image):
        self.image = cv2.imread(image)
        self.find_column(self.image)

    def find_column(self, image):
        # TODO: recode to be automatic
        y, x, c = image.shape
        x_int = np.arange(0, x, math.floor(x / 40))
        column_image = image[math.floor(y/2):y, x_int[21]:x_int[23]]
        cv2.imshow("sanity check", column_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.data = self.horizontal_average(column_image)

    def horizontal_average(self, column_image):
        y, x, c = column_image.shape
        self.scale = x * 10
        column_average = np.zeros(y)
        for i in range(y):
            column_average[i] = np.average(column_image[i, :, 1])
        
        return column_average
    
    def plot(self):
        plt.plot(np.arange(0, len(self.data)), self.data, label="green")
        plt.legend()
        plt.show()

test = preprocess('test_images/cropped_test.jpeg')
test.plot()

# Plot unused code
        # coef = np.polyfit(np.arange(0, len(self.data)), self.data, 1)
        # poly1d_fn = np.poly1d(coef)
        # for i in range(len(self.data)):
        #     if poly1d_fn(i) / 1.05 > self.data[i]:
        #         self.data[i] =  poly1d_fn(i)

        # norm_d = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        # d = np.array([(100, i*255, 0) for i in norm_d])

        # cv2.imshow("green", np.swapaxes(np.array([d for i in range(self.scale)]), 0, 1))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # plt.plot(np.arange(0, len(self.data)), poly1d_fn(np.arange(0, len(self.data))), label='regression')

# Find_column unused code
        # y, x, c = image.shape
        # dx = math.floor(x / 40)
        # x_int = np.arange(0, x, dx)
        # for i in range(len(x_int) - 1):
        #     if i % 5 == 1:
        #         image[:, x_int[i]: x_int[i+1] - math.floor(dx / 2)] = np.array([255, 0, 0])
        #     elif i % 2 == 1:
        #         image[:, x_int[i] : x_int[i+1] - math.floor(dx / 2)] = np.array([0, 0, 0])

        # cv2.imshow("green", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# General unused code
        # image = cv2.imread('test_images/cropped_test.jpeg')
        # y, x, c = image.shape


        # x_int = np.arange(0, x, math.floor(x / 40))

        # for i in range(len(x_int) - 1):
        #     if i == 17 or i == 25:
        #         image[:, x_int[i]:x_int[i+1]] = np.array([0, 0, 0])
        #     if i == 21:
        #         # center_blues = image[math.floor(y / 2):, x_int[i], 0]
        #         center_greens = image[math.floor(y / 2):, x_int[i], 1]
        #         # center_reds = image[math.floor(y / 2):, x_int[i], 2]

        # plt.plot(np.arange(0, y - math.floor(y / 2)), center_greens, label="green")
        # # plt.plot(np.arange(0, y - math.floor(y / 2)), center_blues, label="blue")
        # # plt.plot(np.arange(0, y - math.floor(y / 2)), center_reds, label="red")
        # plt.legend()
        # plt.show()

        # edges = cv2.Canny(image=im12, threshold1=100, threshold2=200) 
        # cv2.imshow("green", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()