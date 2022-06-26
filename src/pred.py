import cv2 as cv
import keyboard

from util import get_eye, get_model, set_cursor

model = get_model()
model.load_weights("weights/eye.h5")

vid = cv.VideoCapture(0)

s_x = []
s_y = []

while True:
    ret, frame = vid.read()
    px, py, eye = get_eye(frame)
    if eye is not None:
        pred = model.predict(eye.reshape((1, 224, 224, 1)), verbose=0)
        pos = pred[0]

        s_x.append(pos[0])
        s_y.append(pos[1])
        if len(s_x) > 3:
            s_x.pop(0)
            s_y.pop(0)
        ax = sum(s_x) / len(s_x)
        ay = sum(s_y) / len(s_y)

        set_cursor([ax, ay])
        print(pos)
        cv.imshow("Eye", eye)
        cv.waitKey(1)
        if keyboard.is_pressed("q"):
            break

vid.release()
