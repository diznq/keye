import cv2 as cv
import keyboard
import numpy as np

from util import get_cursor, get_eye

vid = cv.VideoCapture(0)
X = []
Y = []
Z = []


positions = np.arange(0.0, 1.0, 0.25)

while True:
    ret, frame = vid.read()

    px, py, eye = get_eye(frame)
    if eye is not None:
        pos = get_cursor()
        X.append(eye)
        Y.append(pos)
        Z.append([px, py])
        cv.imshow("Eye", eye)
        cv.waitKey(1)

    if keyboard.is_pressed("q"):
        break

vid.release()

X = np.asarray(X)
Y = np.asarray(Y)
Z = np.asarray(Z)

np.savez("datasets/df.npz", x=X, y=Y, z=Z)
