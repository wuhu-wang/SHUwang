import cv2 as cv
import numpy as np


# 转灰度图、二值图
def show_img(pic):
    _img = cv.imread(pic)
    _gray = cv.cvtColor(_img, cv.COLOR_BGR2GRAY)
    _ret, _binary = cv.threshold(_gray, 130, 255, cv.THRESH_BINARY)
    cv.imwrite('./original.jpg', _img)
    cv.imwrite('./gray.jpg', _gray)
    cv.imwrite('./binary.jpg', _binary)


# 线性点运算
def linear(pic, a, b):
    _img = cv.imread(pic)
    _gray = cv.cvtColor(_img, cv.COLOR_BGR2GRAY)
    _output = _gray * a + b
    # 将灰度值超过255的像素点转化为255
    _row, _col = _output.shape[:2]
    for row in range(_row):
        for col in range(_col):
            _output[row, col] = min(255, _output[row, col])
    # 将图像矩阵的每个点从float->int
    _output = _output.astype(np.uint8)
    cv.imshow(f'a = {a}, b = {b}', _output)
    cv.imwrite(f'./a = {a}, b = {b}.jpg', _output)


def part_linear(pic):
    linear(pic, -1, 0)
    linear(pic, 1.5, 0)
    linear(pic, 0.5, 0)
    linear(pic, 1, 20)
    linear(pic, 1.5, 10)
    linear(pic, 0.5, 20)
