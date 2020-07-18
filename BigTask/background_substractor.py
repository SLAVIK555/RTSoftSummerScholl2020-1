import numpy as np

import cv2

  

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

  

while(1):

    ret, frame = cap.read()

  

    fgmask = fgbg.apply(frame)

   

    cv2.imshow('frame', frame)

    cv2.imshow('fgmask', fgmask)

    #imgray = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
    grframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grframe = cv2.GaussianBlur(grframe, (7, 7), 0)

    ret, thresh = cv2.threshold(grframe, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    cv2.imshow('Fgmask Countors', frame)

  

      

    k = cv2.waitKey(30) & 0xff

    if k == 27:

        break

      

  
cap.release()
cv2.destroyAllWindows()