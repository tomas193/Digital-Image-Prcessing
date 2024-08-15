import cv2
import numpy as np
import time

carpeta_imagenes = 'breakdance'
i = 0
while True:
    cadena = ''
    cad2 = str(i)
    for k in range(0, (5 - len(cad2))): cadena += '0'
    cadena += str(i)
    ruta_imagen = f'{carpeta_imagenes}/' + cadena + '.jpg'
    img = cv2.imread(ruta_imagen)
    img2 = cv2.imread(ruta_imagen)
 
    x, y, ancho, alto = 160, 5, 520, 550
    cut_img = img[y:y + alto, x:x + ancho]
    
    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(cut_img, cv2.COLOR_BGR2HSV)
    
    # Definir un rango más fuerte de color café en HSV
    cafe_bajo = np.array([5, 50, 50])
    cafe_alto = np.array([40, 255, 255])
    
    # Definir un rango más claro de color azul en HSV
    azul_claro_bajo = np.array([90, 50, 150])
    azul_claro_alto = np.array([120, 255, 255])
    
    # Crear máscaras para identificar los píxeles café y azul claro
    mask_cafe = cv2.inRange(hsv, cafe_bajo, cafe_alto)
    mask_azul_claro = cv2.inRange(hsv, azul_claro_bajo, azul_claro_alto)
    
    # Combinar las máscaras
    mask_combined = cv2.bitwise_or(mask_cafe, mask_azul_claro)
    
    # Aplicar la máscara combinada a la imagen original
    combined_colors = cv2.bitwise_and(cut_img, cut_img, mask=mask_combined)

    i += 1
    if i == 70:
        i = 0

    cv2.imshow('Detección de Bordes', combined_colors)
    
    cv2.waitKey(10)
    time.sleep(0.01)

cv2.destroyAllWindows()
