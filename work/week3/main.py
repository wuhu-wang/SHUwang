import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import function as fc

if __name__ == '__main__':
    img = cv.imread('lena.jpeg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fc.gama_equ_process1(gray)
    fc.gauss_process2(gray)
    fc.salt_process3(gray)
    cv.waitKey()
