from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import cv2

f = cv2.imread("picture.png", 0)/255.
h = cv2.imread("note.png", 0)/255.

f -= np.mean(f)
h -= np.mean(h)



correlaciones=[]
for i in range(0,4):
  aux=[]
  h = np.rot90(h)

  corr = signal.correlate2d(f, h, boundary='symm', mode = 'same')
  y, x = np.unravel_index(np.argmax(corr), corr.shape)

  aux.append(np.max(corr))
  aux.append(x)
  aux.append(y)
  aux.append(h)
  correlaciones.append(aux)

imagen_buena=min(correlaciones)
del correlaciones

plt.imshow(imagen_buena[3], cmap = 'gray')
plt.show()

plt.imshow(f, cmap = 'gray')
plt.plot(imagen_buena[1], imagen_buena[2], 'r*', markersize = '15')
plt.show()