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


img = cv.imread("lena.bmp",cv.IMREAD_GRAYSCALE)
img_Downsamping = np.zeros((64,64),dtype=np.uint8)
img_Yokoi = np.zeros((64,64),dtype=np.uint8)


for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if i%8 == 0 and j%8 == 0 and img[i,j] >= 128:
            img_Downsamping[int(i/8),int(j/8)] = 1


for i in range(img_Yokoi.shape[0]):
    for j in range(img_Yokoi.shape[1]):
        img_Yokoi[i,j] = Yokoi(img_Downsamping,i,j)

for i in range(img_Yokoi.shape[0]):
    for j in range(img_Yokoi.shape[1]):
        print(img_Yokoi[i,j],end="")
    print("")


f = open('sample.txt','w')
for i in range(img_Yokoi.shape[0]):
    for j in range(img_Yokoi.shape[1]):
            if img_Yokoi[i,j] == 0 :
                print(" ",end="",file=f)
            else :
                print(img_Yokoi[i,j],end="",file=f)
    print("",file=f)
f.close()

print("finish")