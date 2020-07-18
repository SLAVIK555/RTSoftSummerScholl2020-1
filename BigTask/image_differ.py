
import cv2
import numpy as np
import poly_point_isect as bot

#from __future__ import print_function

img1 = cv2.imread("./images/rect64.jpg")
#img2 = cv2.imread("./images/rect_with_box.jpg")
cap = cv2.VideoCapture(0)

while(1):
	ret, img2 = cap.read()
	#cv2.imshow("camera", img2)

	diff = cv2.absdiff(img1, img2)
	mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

	th = 30
	imask =  mask>th

	canvas = np.zeros_like(img2, np.uint8)
	canvas[imask] = img2[imask]

	#cv2.imshow('cavans', canvas)

	#cv2.waitKey(0)



	#Make more contrast
	#-----Converting image to LAB Color model----------------------------------- 
	lab= cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB)
	#cv2.imshow("lab",lab)
	#cv2.waitKey(0)

	#-----Splitting the LAB image to different channels-------------------------
	l, a, b = cv2.split(lab)
	#cv2.imshow('l_channel', l)
	#cv2.imshow('a_channel', a)
	#cv2.imshow('b_channel', b)
	#cv2.waitKey(0)

	#-----Applying CLAHE to L-channel-------------------------------------------
	clahe = cv2.createCLAHE(clipLimit=90.0, tileGridSize=(1,1))
	cl = clahe.apply(l)
	#cv2.imshow('CLAHE output', cl)
	#cv2.waitKey(0)

	#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
	limg = cv2.merge((cl,a,b))
	#cv2.imshow('limg', limg)
	#cv2.waitKey(0)

	#-----Converting image from LAB Color model to RGB model--------------------
	final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	#cv2.imshow('final', final)
	#cv2.waitKey(0)



	#Drawing lines
	img = final
	#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#kernel_size = 5
	#blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)


	low_threshold = 50
	high_threshold = 150
	edges = cv2.Canny(img, low_threshold, high_threshold)#blur_gray
	#cv2.imshow('lines_edges_from_canny', edges)
	#cv2.waitKey(0)


	rho = 1  # distance resolution in pixels of the Hough grid
	theta = np.pi / 180  # angular resolution in radians of the Hough grid
	threshold = 15  # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 50  # minimum number of pixels making up a line
	max_line_gap = 20  # maximum gap in pixels between connectable line segments
	line_image = np.copy(img) * 0  # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
	#print(lines)

	points = []
	for line in lines:
	    for x1,y1,x2,y2 in line:
	        points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
	        cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),5)

	# Draw the lines on the  image
	lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)


	#for line in lines:
	#    for x1,y1,x2,y2 in line:
	#    	cv2.circle(lines_edges,(x1,y1),1,(0,0,255))
	#    	cv2.circle(lines_edges,(x2,y2),1,(0,0,255))



	#Drawing points with harris corner detector
	operatedImage = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

  
	# изменить тип данных
	# установка 32-битной плавающей запятой

	operatedImage = np.float32(operatedImage)

  
	# применить метод cv2.cornerHarris
	# для определения углов с соответствующими
	# значения в качестве входных параметров

	dest = cv2.cornerHarris(operatedImage, 2, 5, 0.15)

  
	# Результаты отмечены через расширенные углы

	dest = cv2.dilate(dest, None)

  
	# Возвращаясь к исходному изображению,
	# с оптимальным пороговым значением

	final[dest > 0.01 * dest.max()]=[0, 0, 255]

  
	# окно с выводимым изображением с углами

	cv2.imshow('final with Borders', final)
	#cv2.waitKey(0)







	#cv2.imshow('lines_edges', lines_edges)
	#cv2.waitKey(0)



	#Drawing points
	#print (points)
	#cv2.waitKey(0)
	#intersections = bot.isect_segments(points)
	#print (intersections)
	#cv2.waitKey(0)

	#for inter in intersections:
	#    a, b = inter
	#    for i in range(5):
	#        for j in range(5):
	#            lines_edges[int(b) + i, int(a) + j] = [0, 0, 255]

	cv2.imshow('lines_with_points', lines_edges)
	#cv2.waitKey(0)



	if cv2.waitKey(10) == 27: # Клавиша Esc
		break


cap.release()
cv2.destroyAllWindows()



#canny = cv2.Canny(canvas, 100, 50)

#cv2.imshow('Canny', canny)

#cv2.waitKey(0)



#imgray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

#ret, thresh = cv2.threshold(imgray, 127, 255, 0)
#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#cv2.drawContours(canvas, contours, -1, (0,255,0), 3)

#cv2.imshow('canvas contours', canvas)

#cv2.waitKey(0)