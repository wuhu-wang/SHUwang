import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# gama变化＋均衡直方图
def gama_equ_process1(gray):
    _row, _col = gray.shape[:2]
    for row in range(_row):
        for col in range(_col):
            gray[row, col] = 5 * pow(gray[row, col], 0.7)
    equ = cv.equalizeHist(gray)
    cv.imwrite("./gama_gray.jpg", gray)
    cv.imwrite("./equ.jpg", equ)
    plt.subplot(221), plt.hist(gray.ravel(), 256, [0, 256])
    plt.title('Histogram'), plt.axis('off')
    plt.subplot(223), plt.hist(equ.ravel(), 256, [0, 256])
    plt.title('Histogram Equalization'), plt.axis('off')
    plt.show()


# 高斯噪声
def gauss_process2(gray):
    _row, _col = gray.shape[:2]
    for row in range(_row):
        for col in range(_col):
            s = np.random.normal(0, 20)
            gray[row, col] = gray[row, col] + s
            if gray[row, col] > 255:
                gray[row, col] = 255
            elif gray[row, col] < 0:
                gray[row, col] = 0
    blur = cv.blur(gray, (5, 5))
    median = cv.medianBlur(gray, 5)
    gauss = cv.GaussianBlur(gray, (5, 5), 1)
    cv.imwrite("./gauss_gray.jpg", gray)
    cv.imwrite("./gauss_blur.jpg", blur)
    cv.imwrite("./gauss_median.jpg", median)
    cv.imwrite("./gauss_gauss.jpg", gauss)


# 椒盐噪声
def salt_process3(gray):
    prob = 0.05
    thres = 1 - prob
    _row, _col = gray.shape[:2]
    for row in range(_row):
        for col in range(_col):
            s = np.random.rand()
            if s < prob:
                gray[row, col] = 0
            elif s > thres:
                gray[row, col] = 255
    blur = cv.blur(gray, (5, 5))
    median = cv.medianBlur(gray, 5)
    gauss = cv.GaussianBlur(gray, (5, 5), 1)
    cv.imwrite("./salt_gray.jpg", gray)
    cv.imwrite("./salt_blur.jpg", blur)
    cv.imwrite("./salt_median.jpg", median)
    cv.imwrite("./salt_gauss.jpg", gauss)
