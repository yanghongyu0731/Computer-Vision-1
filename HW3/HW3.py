from cv2 import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("lena.bmp")
img_b = np.zeros(img.shape,dtype=np.uint8)
img_c = np.zeros(img.shape,dtype=np.uint8)

histogram = np.zeros((256,3),dtype=int)
histogram_b = np.zeros((256,3),dtype=int)
histogram_c = np.zeros((256,3),dtype=int)

Sk = np.zeros((256,3),dtype=int)


for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        (b,g,r) = img[i,j]
        histogram[b,0] = histogram[b,0]+1
        histogram[g,1] = histogram[g,1]+1
        histogram[r,2] = histogram[r,2]+1
        

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        (b,g,r) = img[i,j]
        b = int(b/3)
        g = int(g/3)
        r = int(r/3)
        img_b[i,j] = (b,g,r)
        histogram_b[b,0] = histogram_b[b,0]+1
        histogram_b[g,1] = histogram_b[g,1]+1
        histogram_b[r,2] = histogram_b[r,2]+1

for i in range(256):
    if i == 0:
        Sk[i,:] = histogram_b[i,:]
    else :
        Sk[i,:] = Sk[i-1,:]+ histogram_b[i,:]
Sk = Sk*255/img.shape[0]/img.shape[1]



for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        (b,g,r) = img_b[i,j]
        b = int(Sk[b,0])
        g = int(Sk[g,1])
        r = int(Sk[r,2])
        img_c[i,j] = (b,g,r)
        histogram_c[b,0] = histogram_c[b,0]+1
        histogram_c[g,1] = histogram_c[g,1]+1
        histogram_c[r,2] = histogram_c[r,2]+1


cv.imshow("before",img)
cv.imshow("hw3_b",img_b)
cv.imshow("hw3_c",img_c)

cv.imwrite("before.jpg",img)
cv.imwrite("hw3_b.jpg",img_b)
cv.imwrite("hw3_c.jpg",img_c)


plt.figure(figsize=(12,4))
new_ticks = np.linspace(0, 255,256)
plt.bar(new_ticks,histogram[:,0])
plt.title("histogram")
plt.xlabel("intensity")
plt.ylabel("count")
plt.figure(figsize=(12,4))
new_ticks = np.linspace(0, 255,256)
plt.bar(new_ticks,histogram_b[:,0])
plt.title("histogram_b")
plt.xlabel("intensity")
plt.ylabel("count")
plt.figure(figsize=(12,4))
new_ticks = np.linspace(0, 255,256)
plt.bar(new_ticks,histogram_c[:,0])
plt.title("histogram_c")
plt.xlabel("intensity")
plt.ylabel("count")
plt.show()