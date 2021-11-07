import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv


# 傅里叶变换频谱(对数)
def magnitude_spectrum(_gray):
    _dft = cv.dft(np.float32(_gray), flags=cv.DFT_COMPLEX_OUTPUT)
    _dft_shift = np.fft.fftshift(_dft)
    _magnitude_spectrum = 20 * np.log(cv.magnitude(_dft_shift[:, :, 0], _dft_shift[:, :, 1]))
    plt.subplot(331), plt.imshow(_magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])


# 低通滤波
def low_pass_filtering(_gray, _rows, _cols, _half_rows, _half_cols):
    x = 25
    _dft = cv.dft(np.float32(_gray), flags=cv.DFT_COMPLEX_OUTPUT)
    _dft_shift = np.fft.fftshift(_dft)
    _low_pass_filtering_mask = np.zeros((_rows, _cols, 2), np.uint8)
    _low_pass_filtering_mask[_half_rows - x:_half_rows + x, _half_cols - x:_half_cols + x] = 1
    _low_pass_filtering_shift = _dft_shift * _low_pass_filtering_mask
    _low_pass_filtering_i_shift = np.fft.ifftshift(_low_pass_filtering_shift)
    _low_pass_filtering_img_back = cv.idft(_low_pass_filtering_i_shift)
    _low_pass_filtering_img_back = cv.magnitude(_low_pass_filtering_img_back[:, :, 0],
                                                _low_pass_filtering_img_back[:, :, 1])
    _dft = cv.dft(np.float32(_low_pass_filtering_img_back), flags=cv.DFT_COMPLEX_OUTPUT)
    _dft_shift = np.fft.fftshift(_dft)
    _magnitude_spectrum = 20 * np.log(cv.magnitude(_dft_shift[:, :, 0], _dft_shift[:, :, 1]))
    plt.subplot(336), plt.imshow(_magnitude_spectrum, cmap='gray')
    plt.title('low_Pass_Filtering Magnitude Spectrum'), plt.axis('off')
    plt.subplot(334), plt.imshow(_low_pass_filtering_img_back, cmap='gray')
    plt.title('Low_Pass_Filtering'), plt.xticks([]), plt.yticks([])


# 高通滤波
def high_pass_filtering_magnitude_spectrum(_gray, _half_rows, _half_cols):
    x = 35
    _dft = np.fft.fft2(_gray)
    _dft_shift = np.fft.fftshift(_dft)
    _dft_shift[_half_rows - x:_half_rows + x, _half_cols - x:_half_cols + x] = 0
    _i_shift = np.fft.ifftshift(_dft_shift)
    _img = np.fft.ifft2(_i_shift)
    _img = np.abs(_img)
    _dft = cv.dft(np.float32(_img), flags=cv.DFT_COMPLEX_OUTPUT)
    _dft_shift = np.fft.fftshift(_dft)
    _magnitude_spectrum = 20 * np.log(cv.magnitude(_dft_shift[:, :, 0], _dft_shift[:, :, 1]))
    plt.subplot(339), plt.imshow(_magnitude_spectrum, cmap='gray')
    plt.title('High_Pass_Filtering Magnitude Spectrum'), plt.axis('off')
    plt.subplot(337), plt.imshow(_img, cmap='gray')
    plt.title('High_Pass_Filtering'), plt.xticks([]), plt.yticks([])