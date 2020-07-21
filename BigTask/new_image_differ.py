# Python-код для поиска координат
# контуры, обнаруженные на изображении.

import numpy as np

import cv2

def corner_find(image):

    # Инициализируем список с точками
    points = []

    # Чтение изображения
    font = cv2.FONT_HERSHEY_COMPLEX
    img2 = image

    # Чтение того же изображения в другом
    # переменная и преобразование в серую шкалу.
    img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  
    # Преобразование изображения в двоичное изображение
    # (только черно-белое изображение).
    _, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

  
    #Обнаружение контуров в изображении.
    contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  
    # Проходя через все контуры, найденные на изображении.
    for cnt in contours :

        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        # рисует границу контуров.
        cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5) 

        # Используется для выравнивания массива, содержащего
        # координаты вершин.
        n = approx.ravel() 
        i = 0

        for j in n :
            if(i % 2 == 0):
                x = n[i]
                y = n[i + 1]

                # Строка, содержащая координаты.
                string = str(x) + " " + str(y) 
                #####cv2.putText(img2, string, (x, y), font, 0.5, (0, 255, 0)) # Отрисовка точек внутри функции
                points.append((x, y))

            i = i + 1

    # Отображение окончательного изображения и возвращаем список с точками.
    #####cv2.imshow('image2', img2) #Отрисовки озображения внутри функции
    #cv2.waitKey(0)

    return points


def histogram_color_dominator(image, k):
    img = image
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k # how many domination color we find (2 is ok)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    #cv2.imshow('res2',res2)
    #cv2.waitKey(0)

    return res2

"""
#img = cv2.imread("./images/rect.jpg")

#histogram_color_dominator(img, 0)
#histogram_color_dominator(img, 1)
#h = histogram_color_dominator(img, 2)
#cv2.imshow('h', h)
#cv2.waitKey(0)
#histogram_color_dominator(img, 4)
#histogram_color_dominator(img, 8)
"""







"""
img = cv2.imread("./images/fig.jpg")
not_func_points = corner_find(img)
print ("Not_func_points:\n {0}".format(not_func_points))

for p in not_func_points:
    cv2.putText(img, str(p), tuple(p), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    cv2.circle(img, tuple(p), 10, (255,0,0), -1)

cv2.imshow('image', img)
cv2.waitKey(0)
# Выход из окна, если на клавиатуре нажата клавиша «q».

#if cv2.waitKey(0) & 0xFF == ord('q'): 

#    cv2.destroyAllWindows()
"""