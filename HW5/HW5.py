from cv2 import cv2 as cv
import numpy as np

def Dilation(image_output,image,ker,ker_num:int,ker_val:int): 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            max_val = 0
            for k in range(kernal_num):
                if i+ker[k,0]>=0 and i+ker[k,0]<image.shape[0] and j+ker[k,1]>=0 and j+ker[k,1]<image.shape[1]:
                    temp = image[i+ker[k,0],j+ker[k,1]]+ker_val
                    if temp > max_val : max_val = temp
            if max_val > 255 : max_val = 255
            image_output[i,j] = max_val
def Erosion(image_output,image,ker,ker_num:int,ker_val:int): 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            min_val = 256
            for k in range(kernal_num):
                if i+ker[k,0]>=0 and i+ker[k,0]<image.shape[0] and j+ker[k,1]>=0 and j+ker[k,1]<image.shape[1]:
                    temp = image[i+ker[k,0],j+ker[k,1]]+ker_val
                    if temp < min_val : min_val = temp
            if min_val < 0 : min_val = 0
            image_output[i,j] = min_val


                
img = cv.imread("lena.bmp",cv.IMREAD_GRAYSCALE)
img_a = np.zeros(img.shape,dtype=np.uint8)
img_b = np.zeros(img.shape,dtype=np.uint8)
img_c = np.zeros(img.shape,dtype=np.uint8)
img_d = np.zeros(img.shape,dtype=np.uint8)

'''
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j]>=128:
            img[i,j] = 255
        else:
            img[i,j] = 0
'''

kernal_num = 3+5+5+5+3
kernal_value = 0
kernal = np.zeros((kernal_num,2),int)
kernal[0,0],kernal[0,1] = -2,-1 
kernal[1,0],kernal[1,1] = -2,0
kernal[2,0],kernal[2,1] = -2,1 
kernal[3,0],kernal[3,1] = -1,-2 
kernal[4,0],kernal[4,1] = -1,-1 
kernal[5,0],kernal[5,1] = -1,0 
kernal[6,0],kernal[6,1] = -1,1 
kernal[7,0],kernal[7,1] = -1,2 
kernal[8,0],kernal[8,1] = 0,-2 
kernal[9,0],kernal[9,1] = 0,-1 
kernal[10,0],kernal[10,1] = 0,0 
kernal[11,0],kernal[11,1] = 0,1 
kernal[12,0],kernal[12,1] = 0,2 
kernal[13,0],kernal[13,1] = 1,-2
kernal[14,0],kernal[14,1] = 1,-1 
kernal[15,0],kernal[15,1] = 1,0 
kernal[16,0],kernal[16,1] = 1,1 
kernal[17,0],kernal[17,1] = 1,2 
kernal[18,0],kernal[18,1] = 2,-1 
kernal[19,0],kernal[19,1] = 2,0 
kernal[20,0],kernal[20,1] = 2,1 
#print(kernal)



print("Start Dilation")
Dilation(img_a,img,kernal,kernal_num,kernal_value)
print("Start Erosion")
Erosion(img_b,img,kernal,kernal_num,kernal_value)
print("Start Opening")
Dilation(img_c,img_b,kernal,kernal_num,kernal_value)
print("Start Closing")
Erosion(img_d,img_a,kernal,kernal_num,kernal_value)


cv.imwrite("Dilation.jpg",img_a)
cv.imwrite("Erosion.jpg",img_b)
cv.imwrite("Opening.jpg",img_c)
cv.imwrite("Closing.jpg",img_d)
cv.waitKey(0)