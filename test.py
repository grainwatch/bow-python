import numpy as np
import cv2 as cv

img = np.zeros((1000, 1000, 1), dtype=np.uint8)
img = cv.rectangle(img, (250, 250), (750, 750), 255, cv.FILLED)
cv.imshow('', img)
cv.waitKey(0)