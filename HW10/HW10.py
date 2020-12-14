from cv2 import cv2 as cv
import numpy as np
import math as m

def zero_crossing(lap, i, j):

    x0 = lap[i, j]

    x1 = x2 = x3 = x4 = x5 = x6 = x7 = x8 = x0

    if i-1 >= 0:
        x2 = float(lap[i-1, j])
    else:
        x2 = x0

    if j-1 >= 0:
        x3 = float(lap[i, j-1])
    else:
        x3 = x0

    if i+1 < lap.shape[0]:
        x4 = float(lap[i+1, j])
    else:
        x4 = x0

    if j+1 < lap.shape[1]:
        x1 = float(lap[i, j+1])
    else:
        x1 = x0

    if i-1 >= 0 and j+1 < lap.shape[1]:
        x6 = float(lap[i-1, j+1])
    elif i-1 >= 0 and j+1 >= lap.shape[1]:
        x6 = x2
    elif i-1 < 0 and j+1 < lap.shape[1]:
        x6 = x1

    if i-1 >= 0 and j-1 >= 0:
        x7 = float(lap[i-1, j-1])
    elif i-1 >= 0 and j-1 < 0:
        x7 = x2
    elif i-1 < 0 and j-1 >= 0:
        x7 = x3

    if j-1 >= 0 and i+1 < lap.shape[0]:
        x8 = float(lap[i+1, j-1])
    elif j-1 < 0 and i+1 < lap.shape[0]:
        x8 = x4
    elif j-1 >= 0 and i+1 >= lap.shape[0]:
        x6 = x3

    if i+1 < lap.shape[0] and j+1 < lap.shape[1]:
        x5 = float(lap[i+1, j+1])
    elif i+1 >= lap.shape[0] and j+1 < lap.shape[1]:
        x5 = x4
    elif i+1 < lap.shape[0] and j+1 >= lap.shape[1]:
        x5 = x1

    if x0 >= 1:
        if min(x1, x2, x3, x4, x5, x6, x7, x8) == -1:
            return 0
        else:
            return 255
    else:
        return 255


def Laplacian(image, i, j, mask, threshold):

    x0 = image[i, j]

    x1 = x2 = x3 = x4 = x5 = x6 = x7 = x8 = x0

    if i-1 >= 0:
        x2 = float(image[i-1, j])
    else:
        x2 = x0

    if j-1 >= 0:
        x3 = float(image[i, j-1])
    else:
        x3 = x0

    if i+1 < image.shape[0]:
        x4 = float(image[i+1, j])
    else:
        x4 = x0

    if j+1 < image.shape[1]:
        x1 = float(image[i, j+1])
    else:
        x1 = x0

    if i-1 >= 0 and j+1 < image.shape[1]:
        x6 = float(image[i-1, j+1])
    elif i-1 >= 0 and j+1 >= image.shape[1]:
        x6 = x2
    elif i-1 < 0 and j+1 < image.shape[1]:
        x6 = x1

    if i-1 >= 0 and j-1 >= 0:
        x7 = float(image[i-1, j-1])
    elif i-1 >= 0 and j-1 < 0:
        x7 = x2
    elif i-1 < 0 and j-1 >= 0:
        x7 = x3

    if j-1 >= 0 and i+1 < image.shape[0]:
        x8 = float(image[i+1, j-1])
    elif j-1 < 0 and i+1 < image.shape[0]:
        x8 = x4
    elif j-1 >= 0 and i+1 >= image.shape[0]:
        x6 = x3

    if i+1 < image.shape[0] and j+1 < image.shape[1]:
        x5 = float(image[i+1, j+1])
    elif i+1 >= image.shape[0] and j+1 < image.shape[1]:
        x5 = x4
    elif i+1 < image.shape[0] and j+1 >= image.shape[1]:
        x5 = x1
    '''
    print(x7,' ',x2,' ',x6)
    print(x3,' ',x0,' ',x1)
    print(x8,' ',x4,' ',x5)
    '''

    sum = mask[0][0]*x7+mask[0][1]*x2+mask[0][2]*x6+mask[1][0]*x3 + \
        mask[1][1]*x0+mask[1][2]*x1+mask[2][0]*x8+mask[2][1]*x4+mask[2][2]*x5
    # print(sum)
    if sum >= threshold:
        return 1
    elif sum <= -threshold:
        return -1
    else:
        return 0


def G_image(image,i,j,mask,threshold):

    s = 0

    for x in range(-5,6):
        for y in range(-5,6):
            s = s + image[i+x,j+y]*mask[x+5][y+5]
    
    if s >= threshold:
        return 1
    elif s <= -threshold:
        return -1
    else:
        return 0
    

def expand(image, size):
    temp = np.zeros(
        (image.shape[0]+2*size, image.shape[1]+2*size), dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp[i+size, j+size] = image[i, j]

    for j in range(image.shape[1]):
        for i in range(size):
            temp[i, j+size] = temp[size, j+size]
            temp[temp.shape[0] - size + i, j +
                 size] = temp[temp.shape[0]-1-size, j+size]

    for i in range(image.shape[1]):
        for j in range(size):
            temp[i+size, j] = temp[i+size, size]
            temp[i+size, temp.shape[1] - size +
                 j] = temp[i+size, temp.shape[1]-1-size]

    for i in range(size):
        for j in range(size):
            temp[i, j] = temp[size, size]
            temp[i+image.shape[0]+size, j] = temp[temp.shape[0] - size - 1, size]
            temp[i, j+image.shape[1]+size] = temp[size, temp.shape[1] - size - 1]
            temp[temp.shape[0]-size+i, temp.shape[0]-size +
                 j] = temp[temp.shape[0]-1-size, temp.shape[1]-size - 1]

    return temp


def DOGmask(mask,sigma1,sigma2):
    a = b = mean = 0
    for i in range(-5,6):
        for j in range(-5,6):
            a = m.exp( -(i*i+j*j)/(2*sigma1*sigma1)) / (m.sqrt(2*m.pi)*sigma1)
            b = m.exp( -(i*i+j*j)/(2*sigma2*sigma2)) / (m.sqrt(2*m.pi)*sigma2)
            mask[i+5][j+5] = a - b
            mean = mean + a - b
    mean = mean / 11 / 11

    for i in range(11):
        for j in range(11):
            mask[i][j] = mask[i][j] - mean

    return 0

img = cv.imread("lena.bmp", cv.IMREAD_GRAYSCALE)
temp = np.zeros((img.shape[0], img.shape[1]), dtype=int)
temp2 = np.zeros((img.shape[0], img.shape[1]), dtype=int)
temp3 = np.zeros((img.shape[0], img.shape[1]), dtype=int)
temp4 = np.zeros((img.shape[0], img.shape[1]), dtype=int)
temp5 = np.zeros((img.shape[0], img.shape[1]), dtype=int)
Laplacian1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
Laplacian2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
Laplacian3 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
LOG = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
DOG = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

mask1 = [[0,  1, 0], [1, -4, 1], [0,  1, 0]]
mask2 = [[1./3,  1./3, 1./3], [1./3, -8./3, 1./3], [1./3,  1./3, 1./3]]
mask3 = [[2./3,  -1./3, 2./3], [-1./3, -4./3, -1./3], [2./3,  -1./3, 2./3]]
mask4 = [[0, 0,  0, -1, -1, -2, -1, -1,  0,  0,  0],
         [0, 0, -2, -4, -8, -9, -8, -4, -2,  0,  0],
         [0, -2, -7, -15, -22, -23, -22, -15, -7, -2,  0],
         [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
         [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
         [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
         [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
         [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
         [0, -2, -7, -15, -22, -23, -22, -15, -7, -2,  0],
         [0, 0, -2, -4, -8, -9, -8, -4, -2,  0,  0],
         [0, 0,  0, -1, -1, -2, -1, -1,  0,  0,  0]]
'''
mask5 = [[-1, -3,  -4, -6, -7, -8, -7, -6,  -4,  -3,  -1],
         [-3, -5, -8, -11, -13, -13, -13, -11, -8,  -5,  -3],
         [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8,  -4],
         [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
         [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
         [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
         [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
         [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
         [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8,  -4],
         [-3, -5, -8, -11, -13, -13, -13, -11, -8,  -5,  -3],
         [-1, -3,  -4, -6, -7, -8, -7, -6,  -4,  -3,  -1]]
'''

mask5 = [[0 for i in range(11)] for j in range(11)]
DOGmask(mask5,1,3)


img_exp = expand(img,5)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        '''
        temp[i, j] = Laplacian(img, i, j, mask1, 15)
        temp2[i, j] = Laplacian(img, i, j, mask2, 15)
        temp3[i, j] = Laplacian(img, i, j, mask3, 20)
        temp4[i, j] = G_image(img_exp, i+5, j+5, mask4, 3000)
        '''
        temp5[i, j] = G_image(img_exp, i+5, j+5, mask5, 1)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        '''
        Laplacian1[i, j] = zero_crossing(temp, i, j)
        Laplacian2[i, j] = zero_crossing(temp2, i, j)
        Laplacian3[i, j] = zero_crossing(temp3, i, j)
        LOG[i, j] = zero_crossing(temp4, i, j)
        '''
        DOG[i, j] = zero_crossing(temp5, i, j)

'''
cv.imwrite('Laplacian1.png', Laplacian1)
cv.imwrite('Laplacian2.png', Laplacian2)
cv.imwrite('Laplacian3.png', Laplacian3)
cv.imwrite('LOG.png', LOG)
'''
cv.imwrite('DOG2.png', DOG)


