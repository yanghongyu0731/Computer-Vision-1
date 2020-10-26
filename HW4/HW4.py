from cv2 import cv2 as cv
import numpy as np

def Dilation(image_output,image,ker,ker_num): 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] == 255:
                for k in range(kernal_num):
                    if i+ker[k,0]>=0 and i+ker[k,0]<image.shape[0] and j+ker[k,1]>=0 and j+ker[k,1]<image.shape[1]:
                        image_output[i+ker[k,0],j+ker[k,1]] = 255
def Erosion(image_output,image,ker,ker_num): 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp = True
            for k in range(ker_num):
                if i+ker[k,0]>=0 and i+ker[k,0]<image.shape[0] and j+ker[k,1]>=0 and j+ker[k,1]<image.shape[1] and image[i+ker[k,0],j+ker[k,1]] == 0: 
                    temp = False
                    break
            if temp == True:
                image_output[i,j] = 255

def Hit_and_Miss(image_output,image,ker1,ker1_num,ker2,ker2_num):
    img_1 = np.zeros(image.shape,dtype=np.uint8)
    img_2 = np.zeros(image.shape,dtype=np.uint8)
    img_3 = np.zeros(image.shape,dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] == 255:
                img_2[i,j] = 0
            else:
                img_2[i,j] = 255

    Erosion(img_1,image,ker1,ker1_num)
    Erosion(img_3,img_2,ker2,ker2_num)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if img_1[i,j] == img_3[i,j] and img_1[i,j]==255:
                image_output[i,j] = 255
            else:
                image_output[i,j] = 0

                
img = cv.imread("lena.bmp",cv.IMREAD_GRAYSCALE)
img_a = np.zeros(img.shape,dtype=np.uint8)
img_b = np.zeros(img.shape,dtype=np.uint8)
img_c = np.zeros(img.shape,dtype=np.uint8)
img_d = np.zeros(img.shape,dtype=np.uint8)
img_e = np.zeros(img.shape,dtype=np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j]>=128:
            img[i,j] = 255
        else:
            img[i,j] = 0

kernal_num = 3+5+5+5+3
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

kernal_l_num = 3
kernal_l = np.zeros((kernal_l_num,2),int)
kernal_l2 = np.zeros((kernal_l_num,2),int)
kernal_l[0,0],kernal_l[0,1] = 0,0 
kernal_l[1,0],kernal_l[1,1] = 0,-1
kernal_l[2,0],kernal_l[2,1] = 1,0 
kernal_l2[0,0],kernal_l2[0,1] = -1,0 
kernal_l2[1,0],kernal_l2[1,1] = -1,1
kernal_l2[2,0],kernal_l2[2,1] = 0,1 

print("Start Dilation")
Dilation(img_a,img,kernal,kernal_num)
print("Start Erosion")
Erosion(img_b,img,kernal,kernal_num)
print("Start Opening")
Dilation(img_c,img_b,kernal,kernal_num)
print("Start Closing")
Erosion(img_d,img_a,kernal,kernal_num)

print("Start HitandMiss")
Hit_and_Miss(img_e,img,kernal_l,kernal_l_num,kernal_l2,kernal_l_num)


cv.imwrite("binarize.jpg",img)
cv.imwrite("Dilation.jpg",img_a)
cv.imwrite("Erosion.jpg",img_b)
cv.imwrite("Opening.jpg",img_c)
cv.imwrite("Closing.jpg",img_d)
cv.imwrite("Hit-and-Miss.jpg",img_e)
cv.waitKey(0)