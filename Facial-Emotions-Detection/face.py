import cv2
import os
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = load_model('oe_model.h5')

def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, color_mode='grayscale', target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Escalar los valores de los píxeles entre 0 y 1
    return img_array

def prediccion():
    img_path = 'output/foto.jpg'

    # Preprocesar la imagen
    target_size = (48, 48)  # El tamo que utilizaste para entrenar el modelo
    img_array = preprocess_image(img_path, target_size)

    # Mostrar la imagen preprocesada
    plt.imshow(img_array[0, :, :, 0], cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    plt.show()

    # Realizar la predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Si tienes nombres de clases, puedes decodificar la predicción
    class_names = ['enojo', 'disgusto', 'miedo', 'felicidad', 'neutral', 'triste', 'sorprendido']
    print(f"Predicción: {class_names[predicted_class]}")

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

def save_frame(frame, x, y, w, h, output_dir='output'):
    # Crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = 'foto.jpg'
    
    # Guardar la imagen en el directorio de salida
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, frame)
    print(f'Imagen guardada en: {output_path}')

while True:
    # Leer un fotograma de la cámara
    ret, frame = cap.read()
    
    # Convertir el fotograma a escala de grises
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    x, y, w, h = 250, 150, 150, 200  # Coordenadas de la esquina superior izquierda (x, y) y ancho (w) y alto (h)

    roi = frame_gris[y:y+h, x:x+w]

    color = (255,0,0)  # Blanco
    thickness = 2  # Grosor del rectángulo
    cv2.rectangle(frame, (x-5, y-5), (x+w+5, y+h+5), color, thickness)

    # Mostrar el fotograma en la ventana
    cv2.imshow('Cámara', frame)
    
    # Si se presiona la tecla 'q', salir del bucle
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key==ord('m'):
        save_frame(roi, x, y, w, h)
        prediccion()

# Liberar la captura de video y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
