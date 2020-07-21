import cv2
import numpy as np
import glob

from new_image_differ import histogram_color_dominator as hcd
from new_image_differ import corner_find as cf

def box_recognizer(image, net, classes, output_layers, required_confidence):
	img = image

	height, width, channels = img.shape

	# Detecting objects
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

	net.setInput(blob)
	outs = net.forward(output_layers)

	# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > required_confidence:
				# Object detected
				print(class_id)
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)

				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)

				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	print(indexes)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]

			crop_img = img[y:y+h, x:x+w]
			hcd_img = hcd(crop_img, 2)
			box_points = cf(hcd_img)
			#label = str(classes[class_ids[i]])
			#color = colors[class_ids[i]]
			#cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			#cv2.putText(img, label, (x, y + 30), font, 3, color, 2)