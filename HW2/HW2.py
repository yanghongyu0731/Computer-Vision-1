from cv2 import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('lena.bmp')
cv.imshow('binary image',img)
img2 = np.zeros([img.shape[0],img.shape[1]],dtype=int)

histogram = np.zeros(256,dtype=int)
#(a) a binary image (threshold at 128)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        (b,g,r) = img[i,j]
        histogram[b] = histogram[b]+1
        if b >= 128 :
            img[i,j] = (255,255,255)
            img2[i,j] = 1
        else :
            img[i,j] = (0,0,0)
cv.imshow('binary image',img)
cv.imwrite('binary_imgae.jpg',img)

#(b) a histogram
new_ticks = np.linspace(0, 255,256)
plt.bar(new_ticks,histogram)
plt.xlabel('Intensity Value')
plt.ylabel('Count')
plt.title('Image histogram')


#(c) connected components(regions with + at centroid, bounding box)
count = 0
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        if img2[i][j] > 0:
            count = count+1
            img2[i][j] = count
            

change = True
time = 0
while change:
    time = time+1
    print("TIME ",time)
    change = 0
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if img2[i,j] != 0 :
                smallest = img2[i,j]
                if i-1 >=0 :
                    if img2[i-1,j] != 0 and img2[i-1,j]<smallest :
                        smallest = img2[i-1,j]
                if j-1>=0 :
                    if img2[i,j-1] != 0 and img2[i,j-1]<smallest :
                        smallest = img2[i,j-1]
    
                if smallest != img2[i,j]:
                    img2[i,j] = smallest
                    change = True
    for i in range(img2.shape[0]-1,-1,-1):
        for j in range(img2.shape[1]-1,-1,-1):
            if img2[i,j] != 0 :
                smallest = img2[i,j]
                if i+1<=img2.shape[0]-1:
                    if img2[i+1,j] != 0 and img2[i+1,j]<smallest :
                        smallest = img2[i+1,j]
                if j+1<=img2.shape[1]-1 :
                    if img2[i,j+1] != 0 and img2[i,j+1]<smallest :
                        smallest = img2[i,j+1]
                
                if smallest != img2[i,j]:
                    img2[i,j] = smallest
                    change = True

area = np.zeros(count+1,dtype=int)


for i in range(img2.shape[0]) :
    for j in range(img2.shape[1]) :
        if img2[i,j] != 0 :
            area[img2[i,j]] = area[img2[i,j]]+1


connected_count = 0
for c in range(area.shape[0]):
    if area[c] >= 500 :
        connected_count = connected_count+1
        hight = -1
        low = img.shape[0]
        left = -1
        right = img.shape[1]
        center_row = 0
        center_col = 0
        for i in range(img2.shape[0]):
            for j in range(img2.shape[1]):
                if img2[i,j] == c:
                    center_row = center_row+i
                    center_col = center_col+j
                    if i>hight:
                        hight = i
                    if i<low:
                        low = i
                    if j>left:
                        left = j
                    if j<right:
                        right = j
        center_row = int(center_row/area[c])
        center_col = int(center_col/area[c])
        cv.rectangle(img,(right,low),(left,hight),(255,0,0),2)
        cv.drawMarker(img,(center_col,center_row),(0,0,255),markerType=cv.MARKER_CROSS,markerSize=20,thickness=2)


print("connected_count = ",connected_count)
cv.imshow('connected components',img)
cv.imwrite('connected components.jpg',img)
plt.show()
