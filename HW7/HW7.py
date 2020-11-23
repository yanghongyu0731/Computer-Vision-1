from cv2 import cv2 as cv
import numpy as np

def Yokoi(image,i:int,j:int):
    
    x0=x1=x2=x3=x4=x5=x6=x7=x8=0

    x0 = image[i,j]

    if x0 == 0:
        return 0

    if i-1 >= 0:
        x2 = image[i-1,j]
        if j-1 >= 0 :
            x7 = image[i-1,j-1]

    if j+1 <= 63:
        x1 = image[i,j+1]
        if i-1 >= 0 :
            x6 = image[i-1,j+1]

    if j-1 >= 0:
        x3 = image[i,j-1]
        if i+1 <= 63 :
            x8 = image[i+1,j-1]

    if i+1 <= 63:
        x4 = image[i+1,j] 
        if j+1 <= 63 :
            x5 = image[i+1,j+1]

    count_q = 0
    count_r = 0

    if x0 == x1 and (x6 != x0 or x2 != x0):
            count_q = count_q+1
    elif x0 == x1 and (x6 == x0 and x2 == x0):
            count_r = count_r+1

    if x0 == x2 and (x7 != x0 or x3 != x0):
            count_q = count_q+1
    elif x0 == x2 and (x7 == x0 and x3 == x0):
            count_r = count_r+1

    if x0 == x3 and (x8 != x0 or x4 != x0):
            count_q = count_q+1
    elif x0 == x3 and (x8 == x0 and x4 == x0):
            count_r = count_r+1

    if x0 == x4 and (x5 != x0 or x1 != x0):
            count_q = count_q+1
    elif x0 == x4 and (x5 == x0 and x1 == x0):
            count_r = count_r+1            

    if count_r == 4 : return 5
    else : return count_q

def PairRelation(image,i:int,j:int):

    x0 = x1 = x2 = x3 = x4 = 0

    x0 = image[i,j]

    if x0 != 1 and x0 != 2 :
        return 0
    
    if x0 != 1:
        return 'q'

    if j+1 <= 63:
        x1 = image[i,j+1]
    if i-1 >= 0:
        x2 = image[i-1,j]
    if j-1 >= 0:
        x3 = image[i,j-1]
    if i+1 <= 63:
        x4 = image[i+1,j] 
    
    count = 0
    if x1 == 1 : 
        count += 1
    if x2 == 1 :
        count += 1
    if x3 == 1 :
        count += 1
    if x4 == 1 :
        count += 1
    
    if count >= 1 : return 'p'
    return 'q'

def thinning(image_d,image_P,i:int,j:int):
    if image_P[i,j] == 'p' :
        if Yokoi(image_d,i,j) == 1 :
            return 0
    return 1

img = cv.imread("lena.bmp",cv.IMREAD_GRAYSCALE)
img_Downsamping = np.zeros((64,64),dtype=np.uint8)
img_Yokoi = np.zeros((64,64),dtype=np.uint8)
img_PairRelation = np.zeros((64,64),dtype=object)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if i%8 == 0 and j%8 == 0 and img[i,j] >= 128:
            img_Downsamping[int(i/8),int(j/8)] = 255


for time in range(7):

    for i in range(img_Yokoi.shape[0]):
        for j in range(img_Yokoi.shape[1]):
            img_Yokoi[i,j] = Yokoi(img_Downsamping,i,j)

    for i in range(img_PairRelation.shape[0]):
        for j in range(img_PairRelation.shape[1]):
            img_PairRelation[i,j] = PairRelation(img_Yokoi,i,j)
        
    for i in range(img_Downsamping.shape[0]):
        for j in range(img_Downsamping.shape[1]):
            if thinning(img_Downsamping,img_PairRelation,i,j) == 0:
                img_Downsamping[i,j] = 0


cv.imwrite('hw7.png',img_Downsamping)

