
import cv2
import numpy as np
import poly_point_isect as bot
from new_image_differ import corner_find as cf

#from __future__ import print_function

img1 = cv2.imread("./images/rect64.jpg")
#img2 = cv2.imread("./images/rect_with_box.jpg")
cap = cv2.VideoCapture(0)

while(1):
	ret, img2 = cap.read()
	#img2 = cv2.imread("./images/fig.jpg")

	#cv2.imshow("camera", img2)

	diff = cv2.absdiff(img1, img2)
	mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

	th = 35
	imask =  mask>th

	canvas = np.zeros_like(img2, np.uint8)
	canvas[imask] = img2[imask]

	cv2.imshow('cavans', canvas)


	#cf(canvas)

	Gaussian = cv2.GaussianBlur(canvas, (7, 7), 0)

	#cv2.waitKey(0)
	"""
	font = cv2.FONT_HERSHEY_COMPLEX

	img3 = canvas
	img = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

	# Преобразование изображения в двоичное изображение
	# (только черно-белое изображение).

	rett, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

  
	#Обнаружение контуров в изображении.

	contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# Проходя через все контуры, найденные на изображении.

	for cnt in contours :
		approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

		# рисует границу контуров.

		cv2.drawContours(img3, [approx], 0, (0, 0, 255), 5)

		# Используется для выравнивания массива, содержащего

		# координаты вершин.

		n = approx.ravel() 
		i = 0

		for j in n :
			if(i % 2 == 0):
				x = n[i]# x - четные, y - нечетные
				y = n[i + 1]# все начинается с нуля

				# Строка, содержащая координаты.

				string = "x:" + str(x) + " " + "y:" + str(y)

				cv2.putText(img3, string, (x, y), font, 0.5, (0, 255, 0))

		i = i + 1

	# Отображение окончательного изображения.

	cv2.imshow('image3', img3) 
	"""








	
	#Make more contrast
	#-----Converting image to LAB Color model----------------------------------- 
	lab= cv2.cvtColor(Gaussian, cv2.COLOR_BGR2LAB)
	#lab2= cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
	#cv2.imshow("lab",lab)
	#cv2.waitKey(0)

	#-----Splitting the LAB image to different channels-------------------------
	l, a, b = cv2.split(lab)
	#l2, a2, b2 = cv2.split(lab2)
	#cv2.imshow('l_channel', l)
	#cv2.imshow('a_channel', a)
	#cv2.imshow('b_channel', b)
	#cv2.waitKey(0)

	#-----Applying CLAHE to L-channel-------------------------------------------
	clahe = cv2.createCLAHE(clipLimit=90.0, tileGridSize=(1,1))
	cl = clahe.apply(l)
	#cl2 = clahe.apply(l2)
	#cv2.imshow('CLAHE output', cl)
	#cv2.waitKey(0)

	#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
	limg = cv2.merge((cl,a,b))
	#limg2 = cv2.merge((cl2,a2,b2))
	#cv2.imshow('limg', limg)
	#cv2.waitKey(0)

	#-----Converting image from LAB Color model to RGB model--------------------
	final_canvas = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	#final2 = cv2.cvtColor(limg2, cv2.COLOR_LAB2BGR)
	cv2.imshow('final', final_canvas)
	#cv2.waitKey(0)
	cf(final_canvas)


	"""
	font = cv2.FONT_HERSHEY_COMPLEX

	img3 = final_canvas
	img = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

	# Преобразование изображения в двоичное изображение
	# (только черно-белое изображение).

	rett, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

  
	#Обнаружение контуров в изображении.

	contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#print (contours)
	# Проходя через все контуры, найденные на изображении.

	for cnt in contours :
		approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

		# рисует границу контуров.

		cv2.drawContours(img3, [approx], 0, (0, 0, 255), 2)

		# Используется для выравнивания массива, содержащего

		# координаты вершин.

		n = approx.ravel() 

		i = 0

		for j in n:

			if(i % 2 == 0):

				x = n[i]

				y = n[i + 1]



			# Строка, содержащая координаты.

			string = str(x) + " " + str(y)

			cv2.putText(img2, string, (x, y), font, 0.5, (0, 255, 0)) 
			
			i = i + 1

	# Отображение окончательного изображения.

	cv2.imshow('image3', img3) 
	"""

	"""
	#Drawing lines
	img = final_canvas
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
	cv2.imshow('lines_with_points', lines_edges)

	#for line in lines:
	#    for x1,y1,x2,y2 in line:
	#    	cv2.circle(lines_edges,(x1,y1),1,(0,0,255))
	#    	cv2.circle(lines_edges,(x2,y2),1,(0,0,255))
	"""

	"""
	#Drawing points with harris corner detector
	operatedImage = cv2.cvtColor(final_canvas, cv2.COLOR_BGR2GRAY)

  
	# изменить тип данных
	# установка 32-битной плавающей запятой

	operatedImage = np.float32(operatedImage)

  
	# применить метод cv2.cornerHarris
	# для определения углов с соответствующими
	# значения в качестве входных параметров

	dest = cv2.cornerHarris(operatedImage, 8, 7, 0.005)

  
	# Результаты отмечены через расширенные углы

	dest = cv2.dilate(dest, None)

  
	# Возвращаясь к исходному изображению,
	# с оптимальным пороговым значением

	final_canvas[dest > 0.10 * dest.max()]=[0, 0, 255]

  
	# окно с выводимым изображением с углами

	#cv2.imshow('final_canvas with Borders', final_canvas)
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

	#cv2.imshow('lines_with_points', lines_edges)
	#cv2.waitKey(0)
	"""


	if cv2.waitKey(10) == 27: # Клавиша Esc
		break

	if cv2.waitKey(100) == 32: #Клавиша Пробел
		img1 = img2





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