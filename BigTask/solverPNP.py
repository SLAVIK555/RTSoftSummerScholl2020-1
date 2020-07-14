#!/usr/bin/env python
 
import cv2
import numpy as np
 
# Read Image
im = cv2.imread("./images/resized_table.jpg");
size = im.shape
     
#2D image points. If you change the image, you need to change vector
image_points = np.array([
                            (39, 588),     # Near left
                            (588, 584),     # Near right
                            (155, 193),     # Far left
                            (418, 193),     # Far right
                        ], dtype="double")
 
# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Near left
                            (0.0, 0.0, 1000.0),        # Near right
                            (-2000.0, 0.0, 0.0),     # Far left
                            (-2000.0, 0.0, 1000.0),      # Far right
                        ])
 
 
# Camera internals
 
focal_length = size[1]
center = (size[1]/2, size[0]/2)
#camera_matrix = np.array(
#                         [[focal_length, 0, center[0]],
#                         [0, focal_length, center[1]],
#                         [0, 0, 1]], dtype = "double"
#                         )
# 
#print ("Camera Matrix :\n {0}".format(camera_matrix))
 
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
print ("Dist Coeff :\n {0}".format(dist_coeffs))

camera_matrix = np.array(
                        [[616.77545938, 0, 320.67141362],
                        [0, 616.77545938, 229.2410102],
                        [0, 0, 1]], dtype = "double"
                        )
 
print ("Camera Matrix :\n {0}".format(camera_matrix))
 
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
cv2.imshow("Output", im)
cv2.waitKey(0)
