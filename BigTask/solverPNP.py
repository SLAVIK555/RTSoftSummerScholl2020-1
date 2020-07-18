#!/usr/bin/env python
 
import cv2
import numpy as np
 
# Read Image
im = cv2.imread("./images/rectangle.jpg");
size = im.shape
     
#2D image points. If you change the image, you need to change vector
image_points = np.array([
                            (161, 438),     # Near left
                            (1160, 474),     # Near right
                            (362, 283),     # Far left
                            (985, 301),     # Far right
                        ], dtype="double")
 
# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Near left
                            (0.0, 0.0, 2000.0),        # Near right
                            (-1000.0, 0.0, 0.0),     # Far left
                            (-1000.0, 0.0, 2000.0),      # Far right
                        ])
 
 
# Camera internals
 
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                        [[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double"
                        )

print ("Camera Matrix :\n {0}".format(camera_matrix))
 
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
print ("Dist Coeff :\n {0}".format(dist_coeffs))

#camera_matrix = np.array(
#                        [[616.77545938, 0, 320.67141362],
#                        [0, 616.77545938, 229.2410102],
#                        [0, 0, 1]], dtype = "double"
#                        )
# 
#print ("Camera Matrix :\n {0}".format(camera_matrix))
 
#dist_coeffs =  [[ -2.10203569e-01]
#                [ 3.46122613e-01]
#                [ 3.42168373e-03]
#                [ 3.28612782e-03]
#                [ 4.25482205e+00]]

#print ("Dist Coeff :\n {0}".format(dist_coeffs))

(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,)
 
print ("Rotation Vector:\n {0}".format(rotation_vector))
print ("Translation Vector:\n {0}".format(translation_vector))
 
 
# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose
 
 
#(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
 
#for p in image_points:
#    cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
 
 
#p1 = ( int(image_points[0][0]), int(image_points[0][1]))
#p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
 
#cv2.line(im, p1, p2, (255,0,0), 2)
 
# Display image
#cv2.imshow("Output", im)
#cv2.waitKey(0)

#,mtx=camera_matrix,dist=dist_coeffs,rvec=rotation_vector,tvec=translation_vector
refPt = []

def mouse_click_event(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),5,(255,0,0),-1)
        #refPt[0] = x
        #refPt[1] = y
        refPt = [(x, y)]

        #rotM = np.zeros((3,3))
        #cv2.Rodrigues(rvec, rotM)
        #print ("rotM :\n {0}".format(rotM))
        #camMat = np.asarray(mtx)
        #iRot = np.linalg.inv(rotM)
        #iCam = np.linalg.inv(camMat)

        print (refPt[0])
        #(x, y) = (refPt[0], refPt[1])
        #new_coord = get_coord(refPt, mtx, dist, iRot, iCam, tvec)
        create_mask_matrix(camera_matrix, dist_coeffs, rotation_vector, translation_vector, refPt)


def create_mask_matrix(mtx, dist, rvec, tvec, pointxy):
    rotM = np.zeros((3,3))
    cv2.Rodrigues(rvec, rotM)
    #print ("rotM :\n {0}".format(rotM))
    camMat = np.asarray(mtx)
    iRot = np.linalg.inv(rotM)
    iCam = np.linalg.inv(camMat)

    #create matrix with coord
    #for (y,x), _ in np.ndenumerate(self.mask):
    #    if self.mask[y][x] == 1:
    #(x, y) = (223, 399)
    #print (refPt[0])
    #(x, y) = (refPt[0], refPt[1])
    new_coord = get_coord(pointxy, mtx, dist, iRot, iCam, tvec)
    #mask_matrix[y][x] = [new_coord[0], new_coord[1]]


def get_coord(point, mtx, dist, iRot, iCam, tvec):
    un_point = cv2.undistortPoints(np.array(point).astype(np.float32), mtx, dist, P=mtx)
    un_point = un_point.ravel().tolist()
    uvPoint = np.matrix([un_point[0], un_point[1], 1]).T
    tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
    tempMat2 = np.matmul(iRot, tvec)
    s = (0 + tempMat2[1, 0]) / tempMat[1, 0]
    wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvec))
    wcPoint[1] = 0
    point3d = wcPoint.T.tolist()[0]
    print ("X: {0}".format(point3d[0]))
    print ("Y: {0}".format(point3d[1]))
    print ("Z: {0}".format(point3d[2]))
    print ("\n")
    #print ("X:\n")
    #print (point3d[0])
    #print ("Z:\n")
    #print (point3d[2])
    return [point3d[0], point3d[2]]

#cv2.namedWindow('image')
#cv2.setMouseCallback('image', mouse_click_event)

#cap = cv2.VideoCapture(0)
#if not cap:
#    print("!!! Failed VideoCapture: invalid parameter!")

#while(True):
#    # Capture frame-by-frame
#    ret, img = cap.read()
#    if type(img) == type(None):
#        print("!!! Couldn't read frame!")
#        break

#    # Display the resulting frame
#    cv2.imshow('image',img)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

# release the capture
#cap.release()
#cv2.destroyAllWindows()


img = cv2.imread('./images/rectangle.jpg')
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_click_event)

while(1):
    #create_mask_matrix(camera_matrix, dist_coeffs, rotation_vector, translation_vector)
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()

create_mask_matrix(camera_matrix, dist_coeffs, rotation_vector, translation_vector)