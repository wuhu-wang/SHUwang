import week2_function as fc

if __name__ == '__main__':
    img = fc.cv.imread('lena.jpeg')
    gray = fc.cv.cvtColor(img, fc.cv.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    half_rows, half_cols = int(rows / 2), int(cols / 2)
    fc.magnitude_spectrum(gray)
    fc.high_pass_filtering_magnitude_spectrum(gray, half_rows, half_cols)
    fc.low_pass_filtering(gray, rows, cols, half_rows, half_cols)
    fc.plt.show()
