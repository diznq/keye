import os
import cv2 as cv
import keyboard
import numpy as np

from util import get_cursor, get_eye

X = []
Y = []
Z = []

Ox, Oy, Oz = (None, None, None)

if os.path.exists("datasets/df.npz"):
    df = np.load("datasets/df.npz")
    Ox = df["x"]
    Oy = df["y"]
    Oz = df["z"]
    df = None

vid = cv.VideoCapture(0)

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

X = np.concatenate([Ox, np.asarray(X)])
Y = np.concatenate([Oy, np.asarray(Y)])
Z = np.concatenate([Oz, np.asarray(Z)])

np.savez("datasets/df.npz", x=X, y=Y, z=Z)
