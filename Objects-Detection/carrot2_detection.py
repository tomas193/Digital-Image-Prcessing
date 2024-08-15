from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import cv2

f = cv2.imread("full.png", 0)/255.
h = cv2.imread("carrot.png", 0)/255.

f -= np.mean(f)
h -= np.mean(h)

correlaciones=[]
for i in range(0,4):
  h = np.rot90(h)

  corr = signal.correlate2d(f, h, boundary='symm', mode = 'same')
  y, x = np.unravel_index(np.argmax(corr), corr.shape)

  plt.imshow(h, cmap = 'gray')
  plt.show()

  plt.imshow(f, cmap = 'gray')
  plt.plot(x, y, 'r*', markersize = '15')
  plt.show()

