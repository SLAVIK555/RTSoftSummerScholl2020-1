#!/usr/bin/env python
 
import cv2
import numpy as np

tab = cv2.imread("./images/rect.jpg")
tabwb = cv2.imread("./images/rect_with_box.jpg")

rez = tabwb - tab

cv2.imshow('rez', rez)

cv2.waitKey(0)

#kernel = np.ones((5,5), np.float32)/75
#print("Kernel:")
#print(kernel)

#rez_blur = cv2.filter2D(rez, -1, kernel)

# Gaussian Blur

Gaussian = cv2.GaussianBlur(rez, (7, 7), 0)

cv2.imshow('Gaussian Blurring', Gaussian)

cv2.waitKey(0)

#rez_edges = cv2.Canny(Gaussian, 100, 200)

#cv2.imshow('rez_edges', rez_edges)
#cv2.waitKey(0)
imgray = cv2.cvtColor(Gaussian, cv2.COLOR_BGR2GRAY)



ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(Gaussian, contours, -1, (0,255,0), 3)

cv2.imshow('Gaussian Countors', Gaussian)

cv2.waitKey(0)



#hsv = cv2.cvtColor(rez, cv2.COLOR_BGR2HSV)

## mask of green (36,0,0) ~ (70, 255,255)
#mask1 = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))

## mask o yellow (15,0,0) ~ (36, 255, 255)
#mask2 = cv2.inRange(hsv, (15,0,0), (36, 255, 255))

## final mask and masked
#mask = cv2.bitwise_or(mask1, mask2)
#target = cv2.bitwise_and(rez, rez, mask=mask)

#cv2.imshow('target', target)

#cv2.waitKey(0)

#cv2.imwrite("target.png", target)

# Gaussian Blur

#Gaussian = cv2.GaussianBlur(rez, (19, 19), 0)

#cv2.imshow('Gaussian Blurring', Gaussian)

#cv2.waitKey(0)

#Canny = cv2.Canny(Gaussian, 100, 200)

#cv2.imshow('Canny', Canny)

#cv2.waitKey(0)