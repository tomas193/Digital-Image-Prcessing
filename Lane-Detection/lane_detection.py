import cv2
import numpy as np

carpeta_imagenes = 'lanepics'

for i in range(1,699):
    cadena=''
    cad2=str(i)
    for k in range(0,(4-len(cad2))):cadena+='0'
    cadena+=str(i)
    ruta_imagen = f'{carpeta_imagenes}/'+cadena+'.jpg'
    img = cv2.imread(ruta_imagen,0)
    img2 = cv2.imread(ruta_imagen)

    x, y, ancho, alto = 200, 360, 220, 60

    cut_img = img[y:y+alto, x:x+ancho]

    bordes = cv2.Canny(cut_img, 100, 150)

    x=0;y=0
    for i in range(len(bordes),0,-1):
        if sum(bordes[i-1])>0:
            x=220+(i-1)
            for k in range(0,len(bordes[i-1])):
                if k>0:y=350+k;break
            break

    vertices = np.array([[x, y], [x+60, y], [x+120, y+50], [x-60, y+50]], np.int32)
    vertices = vertices.reshape((-1, 1, 2))
    color = (0, 255, 0)
    cv2.fillPoly(img2, [vertices], color=color)

    vertices = np.array([[x-80, y], [x, y], [x-60, y+50], [x-200, y+20]], np.int32)
    vertices = vertices.reshape((-1, 1, 2))
    color = (0, 0, 255)
    cv2.fillPoly(img2, [vertices], color=color)

    cv2.imshow('Detección de Bordes', img2)
    cv2.waitKey(10)

cv2.destroyAllWindows()