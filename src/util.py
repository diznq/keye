from ctypes import Structure, byref, c_long, windll
from time import perf_counter

import cv2 as cv
import win32api
from tensorflow.keras import layers, models

user32 = windll.user32
screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]


cls = cv.CascadeClassifier("assets/haarcascade_eye.xml")
eye_x = []
eye_y = []


def get_eye(frame):
    global eye_x
    global eye_y
    global eye_n

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    eyes = cls.detectMultiScale(frame)

    eyes = sorted(eyes, key=lambda x: (x[0], x[1]))
    start = perf_counter()

    for eye in eyes:
        eye = eyes[len(eyes) - 1]
        sx = eye[0]
        sy = eye[1]
        ex = sx + eye[2]
        ey = sy + eye[3]

        px = (0.5 * sx + 0.5 * ex) / frame.shape[1]
        py = (0.5 * sy + 0.5 * ey) / frame.shape[0]

        if len(eye_x) == 0 or len(eye_y) == 0:
            eye_x.append(px)
            eye_y.append(py)

        avg_x = sum(eye_x) / len(eye_x)
        avg_y = sum(eye_y) / len(eye_y)

        dst = ((avg_x - px) ** 2 + (avg_y - py) ** 2) ** 0.5

        if dst > 0.1:
            continue

        ax = int(avg_x * frame.shape[1])
        ay = int(avg_y * frame.shape[0])
        off = int((frame.shape[1] * frame.shape[0]) ** 0.5) // 12
        off_x = off
        off_y = off
        sx = ax - off_x // 2
        sy = ay - off_y // 2
        ex = sx + off_x
        ey = sy + off_y

        eye = frame[sy:ey, sx:ex]
        eye = cv.fastNlMeansDenoising(eye)
        before = cv.equalizeHist(eye)
        eye = before // 2 + eye // 2
        eye = cv.resize(eye, (224, 224)) * 1.0 / 255
        
        eye_x.append(px)
        eye_y.append(py)

        if len(eye_x) > 10:
            eye_x.pop(0)
            eye_y.pop(0)
        print(perf_counter() - start)
        return px, py, eye

    return 0, 0, None


def get_cursor():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return [pt.x / screen_size[0], pt.y / screen_size[1]]


def set_cursor(pos):
    x = int(pos[0] * screen_size[0])
    y = int(pos[1] * screen_size[1])
    win32api.SetCursorPos((x, y))


def get_model():
    model = models.Sequential()
    shape = (224, 224, 1)
    for i in range(0, 6):
        if i == 0:
            model.add(
                layers.Conv2D(
                    32 + 2 ** (4 + i),
                    (3, 3),
                    strides=(2, 2),
                    activation="selu",
                    input_shape=shape,
                )
            )
        else:
            model.add(
                layers.Conv2D(
                    32 + 2 ** (4 + i), (3, 3), strides=(2, 2), activation="relu"
                )
            )

    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation="relu"))

    model.summary()

    model.compile(optimizer="adam", loss="mse", metrics="mse")
    return model
