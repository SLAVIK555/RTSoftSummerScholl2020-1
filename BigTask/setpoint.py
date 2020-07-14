import cv2
import numpy as np
# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),10,(255,0,0),-1)
        mouseX,mouseY = x,y
        print ("mousex\n")
        print (mouseX)
        print ("mousey\n")
        print (mouseY)


# Create a black image, a window and bind the function to window
img = cv2.imread('./images/resized_table.jpg')
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()