#!/usr/bin/env python
 
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap:
    print("!!! Failed VideoCapture: invalid parameter!")

while(True):
    # Capture frame-by-frame
    ret, current_frame = cap.read()
    if type(current_frame) == type(None):
        print("!!! Couldn't read frame!")
        break

    # Display the resulting frame
    cv2.imshow('frame',current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the capture
cap.release()
cv2.destroyAllWindows()