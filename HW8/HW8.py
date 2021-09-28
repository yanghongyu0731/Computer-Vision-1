from cv2 import cv2 as cv
import numpy as np
import random
from statistics import mean
from statistics import median
import math


def GetGaussianNoise_Image(original_Image, amp):

    gaussianNoise_Image = np.zeros(
        (original_Image.shape[0], original_Image.shape[1]), dtype=np.uint8)
    for r in range(original_Image.shape[0]):
        for c in range(original_Image.shape[1]):
            noisePixel = int(original_Image[r, c]+amp*random.gauss(0, 1))
            if noisePixel > 255:
                noisePixel = 255
            gaussianNoise_Image[r, c] = noisePixel

    return gaussianNoise_Image


def GetSaltAndPepper_Image(original_Image, thr):

    saltandpepper_Image = np.zeros(
        (original_Image.shape[0], original_Image.shape[1]), dtype=np.uint8)
    for r in range(original_Image.shape[0]):
        for c in range(original_Image.shape[1]):
            random_value = random.uniform(0, 1)
            if random_value <= thr:
                saltandpepper_Image[r, c] = 0
            elif random_value >= 1-thr:
                saltandpepper_Image[r, c] = 255
            else:
                saltandpepper_Image[r, c] = original_Image[r, c]

    return saltandpepper_Image


def boxFilter3by3(image, i, j):
    x_set = []
    x0 = image[i, j]
    x_set.append(x0)
    if i-1 >= 0:
        x2 = image[i-1, j]
        x_set.append(x2)
        if j-1 >= 0:
            x7 = image[i-1, j-1]
            x_set.append(x7)

    if j+1 <= 511:
        x1 = image[i, j+1]
        x_set.append(x1)
        if i-1 >= 0:
            x6 = image[i-1, j+1]
            x_set.append(x6)

    if j-1 >= 0:
        x3 = image[i, j-1]
        x_set.append(x3)
        if i+1 <= 511:
            x8 = image[i+1, j-1]
            x_set.append(x8)

    if i+1 <= 511:
        x4 = image[i+1, j]
        x_set.append(x4)
        if j+1 <= 511:
            x5 = image[i+1, j+1]
            x_set.append(x5)

    return mean(x_set)


def boxFilter5by5(image, i, j):
    x_set = []
    x_set.append(image[i, j])
    if i-1 >= 0:
        x_set.append(image[i-1, j])
        if j-1 >= 0:
            x_set.append(image[i-1, j-1])

    if j+1 <= 511:
        x_set.append(image[i, j+1])
        if i-1 >= 0:
            x_set.append(image[i-1, j+1])

    if j-1 >= 0:
        x_set.append(image[i, j-1])
        if i+1 <= 511:
            x_set.append(image[i+1, j-1])

    if i+1 <= 511:
        x_set.append(image[i+1, j])
        if j+1 <= 511:
            x_set.append(image[i+1, j+1])

    ##

    if i-2 >= 0:
        x_set.append(image[i-2, j])
        if j-2 >= 0:
            x_set.append(image[i-2, j-2])

    if j+2 <= 511:
        x_set.append(image[i, j+2])
        if i-2 >= 0:
            x_set.append(image[i-2, j+2])

    if j-2 >= 0:
        x_set.append(image[i, j-2])
        if i+2 <= 511:
            x_set.append(image[i+2, j-2])

    if i+2 <= 511:
        x_set.append(image[i+2, j])
        if j+2 <= 511:
            x_set.append(image[i+2, j+2])

    return mean(x_set)


def medFilter3by3(image, i, j):
    x_set = []
    x0 = image[i, j]
    x_set.append(x0)
    if i-1 >= 0:
        x2 = image[i-1, j]
        x_set.append(x2)
        if j-1 >= 0:
            x7 = image[i-1, j-1]
            x_set.append(x7)

    if j+1 <= 511:
        x1 = image[i, j+1]
        x_set.append(x1)
        if i-1 >= 0:
            x6 = image[i-1, j+1]
            x_set.append(x6)

    if j-1 >= 0:
        x3 = image[i, j-1]
        x_set.append(x3)
        if i+1 <= 511:
            x8 = image[i+1, j-1]
            x_set.append(x8)

    if i+1 <= 511:
        x4 = image[i+1, j]
        x_set.append(x4)
        if j+1 <= 511:
            x5 = image[i+1, j+1]
            x_set.append(x5)
    x_set.sort()
    return x_set[int(len(x_set)/2)]


def medFilter5by5(image, i, j):
    x_set = []
    x0 = image[i, j]
    x_set.append(x0)
    if i-1 >= 0:
        x2 = image[i-1, j]
        x_set.append(x2)
        if j-1 >= 0:
            x7 = image[i-1, j-1]
            x_set.append(x7)

    if j+1 <= 511:
        x1 = image[i, j+1]
        x_set.append(x1)
        if i-1 >= 0:
            x6 = image[i-1, j+1]
            x_set.append(x6)

    if j-1 >= 0:
        x3 = image[i, j-1]
        x_set.append(x3)
        if i+1 <= 511:
            x8 = image[i+1, j-1]
            x_set.append(x8)

    if i+1 <= 511:
        x4 = image[i+1, j]
        x_set.append(x4)
        if j+1 <= 511:
            x5 = image[i+1, j+1]
            x_set.append(x5)

    ##

    if i-2 >= 0:
        x_set.append(image[i-2, j])
        if j-2 >= 0:
            x_set.append(image[i-2, j-2])

    if j+2 <= 511:
        x_set.append(image[i, j+2])
        if i-2 >= 0:
            x_set.append(image[i-2, j+2])

    if j-2 >= 0:
        x_set.append(image[i, j-2])
        if i+2 <= 511:
            x_set.append(image[i+2, j-2])

    if i+2 <= 511:
        x_set.append(image[i+2, j])
        if j+2 <= 511:
            x_set.append(image[i+2, j+2])

    x_set.sort()
    return x_set[int(len(x_set)/2)]


def Dilation(image_output, image, ker, ker_num: int, ker_val: int):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            max_val = 0
            for k in range(ker_num):
                if i+ker[k, 0] >= 0 and i+ker[k, 0] < image.shape[0] and j+ker[k, 1] >= 0 and j+ker[k, 1] < image.shape[1]:
                    temp = image[i+ker[k, 0], j+ker[k, 1]]+ker_val
                    if temp > max_val:
                        max_val = temp
            if max_val > 255:
                max_val = 255
            image_output[i, j] = max_val


def Erosion(image_output, image, ker, ker_num: int, ker_val: int):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            min_val = 256
            for k in range(ker_num):
                if i+ker[k, 0] >= 0 and i+ker[k, 0] < image.shape[0] and j+ker[k, 1] >= 0 and j+ker[k, 1] < image.shape[1]:
                    temp = image[i+ker[k, 0], j+ker[k, 1]]+ker_val
                    if temp < min_val:
                        min_val = temp
            if min_val < 0:
                min_val = 0
            image_output[i, j] = min_val


def SNR(image, clean):
    binarize = np.zeros((image.shape[0], image.shape[1]))
    binarize_c = np.zeros((image.shape[0], image.shape[1]))
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):

            binarize[r, c] = float(image[r, c]/255)
            binarize_c[r, c] = float(clean[r, c]/255)

    mu = np.mean(binarize_c)
    mu_n = np.mean(binarize-binarize_c)

    VS = 0.
    VN = 0.
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            VS = VS + (binarize_c[r, c]-mu)**2
            VN = VN + (binarize[r, c]-binarize_c[r, c]-mu_n)**2

    VN = float(VN/(image.shape[0]/image.shape[1]))
    VS = float(VS/(image.shape[0]/image.shape[1]))

    return 10*math.log10(VS/VN)


# load lena
img = cv.imread("lena.bmp", cv.IMREAD_GRAYSCALE)
rows = img.shape[0]
cols = img.shape[1]

# generate noise image
print('generate noise image')
img_GaussianNoise_10 = GetGaussianNoise_Image(img, 10)
img_GaussianNoise_30 = GetGaussianNoise_Image(img, 30)
img_SaltAndPepper_005 = GetSaltAndPepper_Image(img, 0.05)
img_SaltAndPepper_01 = GetSaltAndPepper_Image(img, 0.1)
# output
cv.imwrite('img_GaussianNoise_10.png', img_GaussianNoise_10)
cv.imwrite('img_GaussianNoise_30.png', img_GaussianNoise_30)
cv.imwrite('img_SaltAndPepper_005.png', img_SaltAndPepper_005)
cv.imwrite('img_SaltAndPepper_01.png', img_SaltAndPepper_01)


# run box filter on all noisy images
print('run box filter on all noisy images')
img_boxFilter3by3_GN10 = np.zeros((rows, cols), np.uint8)
img_boxFilter3by3_GN30 = np.zeros((rows, cols), np.uint8)
img_boxFilter3by3_SAP005 = np.zeros((rows, cols), np.uint8)
img_boxFilter3by3_SAP01 = np.zeros((rows, cols), np.uint8)
img_boxFilter5by5_GN10 = np.zeros((rows, cols), np.uint8)
img_boxFilter5by5_GN30 = np.zeros((rows, cols), np.uint8)
img_boxFilter5by5_SAP005 = np.zeros((rows, cols), np.uint8)
img_boxFilter5by5_SAP01 = np.zeros((rows, cols), np.uint8)
for r in range(rows):
    for c in range(cols):
        img_boxFilter3by3_GN10[r, c] = boxFilter3by3(
            img_GaussianNoise_10, r, c)
        img_boxFilter3by3_GN30[r, c] = boxFilter3by3(
            img_GaussianNoise_30, r, c)
        img_boxFilter3by3_SAP005[r, c] = boxFilter3by3(
            img_SaltAndPepper_005, r, c)
        img_boxFilter3by3_SAP01[r, c] = boxFilter3by3(
            img_SaltAndPepper_01, r, c)
        img_boxFilter5by5_GN10[r, c] = boxFilter5by5(
            img_GaussianNoise_10, r, c)
        img_boxFilter5by5_GN30[r, c] = boxFilter5by5(
            img_GaussianNoise_30, r, c)
        img_boxFilter5by5_SAP005[r, c] = boxFilter5by5(
            img_SaltAndPepper_005, r, c)
        img_boxFilter5by5_SAP01[r, c] = boxFilter5by5(
            img_SaltAndPepper_01, r, c)
# output
cv.imwrite('img_boxFilter3by3_GN10.png', img_boxFilter3by3_GN10)
cv.imwrite('img_boxFilter3by3_GN30.png', img_boxFilter3by3_GN30)
cv.imwrite('img_boxFilter3by3_SAP005.png', img_boxFilter3by3_SAP005)
cv.imwrite('img_boxFilter3by3_SAP01.png', img_boxFilter3by3_SAP01)
cv.imwrite('img_boxFilter5by5_GN10.png', img_boxFilter5by5_GN10)
cv.imwrite('img_boxFilter5by5_GN30.png', img_boxFilter5by5_GN30)
cv.imwrite('img_boxFilter5by5_SAP005.png', img_boxFilter5by5_SAP005)
cv.imwrite('img_boxFilter5by5_SAP01.png', img_boxFilter5by5_SAP01)

# Run median filter on all noisy images
print('run median filter on all noisy images')
img_medFilter3by3_GN10 = np.zeros((rows, cols), np.uint8)
img_medFilter3by3_GN30 = np.zeros((rows, cols), np.uint8)
img_medFilter3by3_SAP005 = np.zeros((rows, cols), np.uint8)
img_medFilter3by3_SAP01 = np.zeros((rows, cols), np.uint8)
img_medFilter5by5_GN10 = np.zeros((rows, cols), np.uint8)
img_medFilter5by5_GN30 = np.zeros((rows, cols), np.uint8)
img_medFilter5by5_SAP005 = np.zeros((rows, cols), np.uint8)
img_medFilter5by5_SAP01 = np.zeros((rows, cols), np.uint8)
for r in range(rows):
    for c in range(cols):
        img_medFilter3by3_GN10[r, c] = medFilter3by3(
            img_GaussianNoise_10, r, c)
        img_medFilter3by3_GN30[r, c] = medFilter3by3(
            img_GaussianNoise_30, r, c)
        img_medFilter3by3_SAP005[r, c] = medFilter3by3(
            img_SaltAndPepper_005, r, c)
        img_medFilter3by3_SAP01[r, c] = medFilter3by3(
            img_SaltAndPepper_01, r, c)
        img_medFilter5by5_GN10[r, c] = medFilter5by5(
            img_GaussianNoise_10, r, c)
        img_medFilter5by5_GN30[r, c] = medFilter5by5(
            img_GaussianNoise_30, r, c)
        img_medFilter5by5_SAP005[r, c] = medFilter5by5(
            img_SaltAndPepper_005, r, c)
        img_medFilter5by5_SAP01[r, c] = medFilter5by5(
            img_SaltAndPepper_01, r, c)
# output
cv.imwrite('img_medFilter3by3_GN10.png', img_medFilter3by3_GN10)
cv.imwrite('img_medFilter3by3_GN30.png', img_medFilter3by3_GN30)
cv.imwrite('img_medFilter3by3_SAP005.png', img_medFilter3by3_SAP005)
cv.imwrite('img_medFilter3by3_SAP01.png', img_medFilter3by3_SAP01)
cv.imwrite('img_medFilter5by5_GN10.png', img_medFilter5by5_GN10)
cv.imwrite('img_medFilter5by5_GN30.png', img_medFilter5by5_GN30)
cv.imwrite('img_medFilter5by5_SAP005.png', img_medFilter5by5_SAP005)
cv.imwrite('img_medFilter5by5_SAP01.png', img_medFilter5by5_SAP01)

# Use both opening-then-closing and closing-then opening filter
kernal_num = 3+5+5+5+3
kernal_value = 0
kernal = np.zeros((kernal_num, 2), int)
kernal[0, 0], kernal[0, 1] = -2, -1
kernal[1, 0], kernal[1, 1] = -2, 0
kernal[2, 0], kernal[2, 1] = -2, 1
kernal[3, 0], kernal[3, 1] = -1, -2
kernal[4, 0], kernal[4, 1] = -1, -1
kernal[5, 0], kernal[5, 1] = -1, 0
kernal[6, 0], kernal[6, 1] = -1, 1
kernal[7, 0], kernal[7, 1] = -1, 2
kernal[8, 0], kernal[8, 1] = 0, -2
kernal[9, 0], kernal[9, 1] = 0, -1
kernal[10, 0], kernal[10, 1] = 0, 0
kernal[11, 0], kernal[11, 1] = 0, 1
kernal[12, 0], kernal[12, 1] = 0, 2
kernal[13, 0], kernal[13, 1] = 1, -2
kernal[14, 0], kernal[14, 1] = 1, -1
kernal[15, 0], kernal[15, 1] = 1, 0
kernal[16, 0], kernal[16, 1] = 1, 1
kernal[17, 0], kernal[17, 1] = 1, 2
kernal[18, 0], kernal[18, 1] = 2, -1
kernal[19, 0], kernal[19, 1] = 2, 0
kernal[20, 0], kernal[20, 1] = 2, 1

img_OpeningThenClosing_GN10 = np.zeros((rows, cols), np.uint8)
img_OpeningThenClosing_GN30 = np.zeros((rows, cols), np.uint8)
img_OpeningThenClosing_SAP005 = np.zeros((rows, cols), np.uint8)
img_OpeningThenClosing_SAP01 = np.zeros((rows, cols), np.uint8)
img_ClosingThenOpening_GN10 = np.zeros((rows, cols), np.uint8)
img_ClosingThenOpening_GN30 = np.zeros((rows, cols), np.uint8)
img_ClosingThenOpening_SAP005 = np.zeros((rows, cols), np.uint8)
img_ClosingThenOpening_SAP01 = np.zeros((rows, cols), np.uint8)
temp1 = np.zeros((rows, cols), np.uint8)
temp2 = np.zeros((rows, cols), np.uint8)
temp3 = np.zeros((rows, cols), np.uint8)

print('OpeningThenClosing')
print('1')
Erosion(temp1, img_GaussianNoise_10, kernal, kernal_num, kernal_value)
Dilation(temp2, temp1, kernal, kernal_num, kernal_value)
Dilation(temp3, temp2, kernal, kernal_num, kernal_value)
Erosion(img_OpeningThenClosing_GN10, temp3, kernal, kernal_num, kernal_value)
print('2')
Erosion(temp1, img_GaussianNoise_30, kernal, kernal_num, kernal_value)
Dilation(temp2, temp1, kernal, kernal_num, kernal_value)
Dilation(temp3, temp2, kernal, kernal_num, kernal_value)
Erosion(img_OpeningThenClosing_GN30, temp3, kernal, kernal_num, kernal_value)
print('3')
Erosion(temp1, img_SaltAndPepper_005, kernal, kernal_num, kernal_value)
Dilation(temp2, temp1, kernal, kernal_num, kernal_value)
Dilation(temp3, temp2, kernal, kernal_num, kernal_value)
Erosion(img_OpeningThenClosing_SAP005, temp3, kernal, kernal_num, kernal_value)
print('4')
Erosion(temp1, img_SaltAndPepper_01, kernal, kernal_num, kernal_value)
Dilation(temp2, temp1, kernal, kernal_num, kernal_value)
Dilation(temp3, temp2, kernal, kernal_num, kernal_value)
Erosion(img_OpeningThenClosing_SAP01, temp3, kernal, kernal_num, kernal_value)

print('ClosingThenOpening')
print('1')
Dilation(temp1, img_GaussianNoise_10, kernal, kernal_num, kernal_value)
Erosion(temp2, temp1, kernal, kernal_num, kernal_value)
Erosion(temp3, temp2, kernal, kernal_num, kernal_value)
Dilation(img_ClosingThenOpening_GN10, temp3, kernal, kernal_num, kernal_value)
print('2')
Dilation(temp1, img_GaussianNoise_30, kernal, kernal_num, kernal_value)
Erosion(temp2, temp1, kernal, kernal_num, kernal_value)
Erosion(temp3, temp2, kernal, kernal_num, kernal_value)
Dilation(img_ClosingThenOpening_GN30, temp3, kernal, kernal_num, kernal_value)
print('3')
Dilation(temp1, img_SaltAndPepper_005, kernal, kernal_num, kernal_value)
Erosion(temp2, temp1, kernal, kernal_num, kernal_value)
Erosion(temp3, temp2, kernal, kernal_num, kernal_value)
Dilation(img_ClosingThenOpening_SAP005, temp3,
         kernal, kernal_num, kernal_value)
print('4')
Dilation(temp1, img_SaltAndPepper_01, kernal, kernal_num, kernal_value)
Erosion(temp2, temp1, kernal, kernal_num, kernal_value)
Erosion(temp3, temp2, kernal, kernal_num, kernal_value)
Dilation(img_ClosingThenOpening_SAP01, temp3, kernal, kernal_num, kernal_value)


cv.imwrite('img_OpeningThenClosing_GN10.png', img_OpeningThenClosing_GN10)
cv.imwrite('img_OpeningThenClosing_GN30.png', img_OpeningThenClosing_GN30)
cv.imwrite('img_OpeningThenClosing_SAP005.png', img_OpeningThenClosing_SAP005)
cv.imwrite('img_OpeningThenClosing_SAP01.png', img_OpeningThenClosing_SAP01)
cv.imwrite('img_ClosingThenOpening_GN10.png', img_ClosingThenOpening_GN10)
cv.imwrite('img_ClosingThenOpening_GN30.png', img_ClosingThenOpening_GN30)
cv.imwrite('img_ClosingThenOpening_SAP005.png', img_ClosingThenOpening_SAP005)
cv.imwrite('img_ClosingThenOpening_SAP01.png', img_ClosingThenOpening_SAP01)


f = open('output.txt', 'w')
img = cv.imread("lena.bmp", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("img_GaussianNoise_10.png", cv.IMREAD_GRAYSCALE)
print('img_GaussianNoise_10 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_GaussianNoise_30.png", cv.IMREAD_GRAYSCALE)
print('img_GaussianNoise_30 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_SaltAndPepper_005.png", cv.IMREAD_GRAYSCALE)
print('img_SaltAndPepper_005 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_SaltAndPepper_01.png", cv.IMREAD_GRAYSCALE)
print('img_SaltAndPepper_01 SNR = ', SNR(img2, img), file=f)


img2 = cv.imread("img_boxFilter3by3_GN10.png", cv.IMREAD_GRAYSCALE)
print('img_boxFilter3by3_GN10 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_boxFilter3by3_GN30.png", cv.IMREAD_GRAYSCALE)
print('img_boxFilter3by3_GN30 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_boxFilter3by3_SAP005.png", cv.IMREAD_GRAYSCALE)
print('img_boxFilter3by3_SAP005 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_boxFilter3by3_SAP01.png", cv.IMREAD_GRAYSCALE)
print('img_boxFilter3by3_SAP01 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_boxFilter5by5_GN10.png", cv.IMREAD_GRAYSCALE)
print('img_boxFilter5by5_GN10 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_boxFilter5by5_GN30.png", cv.IMREAD_GRAYSCALE)
print('img_boxFilter5by5_GN30 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_boxFilter5by5_SAP005.png", cv.IMREAD_GRAYSCALE)
print('img_boxFilter5by5_SAP005 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_boxFilter5by5_SAP01.png", cv.IMREAD_GRAYSCALE)
print('img_boxFilter5by5_SAP01 SNR = ', SNR(img2, img), file=f)


img2 = cv.imread("img_medFilter3by3_GN10.png", cv.IMREAD_GRAYSCALE)
print('img_medFilter3by3_GN10 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_medFilter3by3_GN30.png", cv.IMREAD_GRAYSCALE)
print('img_medFilter3by3_GN30 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_medFilter3by3_SAP005.png", cv.IMREAD_GRAYSCALE)
print('img_medFilter3by3_SAP005 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_medFilter3by3_SAP01.png", cv.IMREAD_GRAYSCALE)
print('img_medFilter3by3_SAP01 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_medFilter5by5_GN10.png", cv.IMREAD_GRAYSCALE)
print('img_medFilter5by5_GN10 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_medFilter5by5_GN30.png", cv.IMREAD_GRAYSCALE)
print('img_medFilter5by5_GN30 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_medFilter5by5_SAP005.png", cv.IMREAD_GRAYSCALE)
print('img_medFilter5by5_SAP005 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_medFilter5by5_SAP01.png", cv.IMREAD_GRAYSCALE)
print('img_medFilter5by5_SAP01 SNR = ', SNR(img2, img), file=f)

img2 = cv.imread("img_OpeningThenClosing_GN10.png", cv.IMREAD_GRAYSCALE)
print('img_OpeningThenClosing_GN10 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_OpeningThenClosing_GN30.png", cv.IMREAD_GRAYSCALE)
print('img_OpeningThenClosing_GN30 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_OpeningThenClosing_SAP005.png", cv.IMREAD_GRAYSCALE)
print('img_OpeningThenClosing_SAP005 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_OpeningThenClosing_SAP01.png", cv.IMREAD_GRAYSCALE)
print('img_OpeningThenClosing_SAP01 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_ClosingThenOpening_GN10.png", cv.IMREAD_GRAYSCALE)
print('img_ClosingThenOpening_GN10 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_ClosingThenOpening_GN30.png", cv.IMREAD_GRAYSCALE)
print('img_ClosingThenOpening_GN30 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_ClosingThenOpening_SAP005.png", cv.IMREAD_GRAYSCALE)
print('img_ClosingThenOpening_SAP005 SNR = ', SNR(img2, img), file=f)
img2 = cv.imread("img_ClosingThenOpening_SAP01.png", cv.IMREAD_GRAYSCALE)
print('img_ClosingThenOpening_SAP01 SNR = ', SNR(img2, img), file=f)

f.closed
