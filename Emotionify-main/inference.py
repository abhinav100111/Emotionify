from pathlib import Path
import os

import cv2
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")

from keras.models import load_model

from emotionify_runtime import load_mediapipe_runtime

BASE_DIR = Path(__file__).resolve().parent

model = load_model(BASE_DIR / "model.h5")
label = np.load(BASE_DIR / "labels.npy", allow_pickle=True)
holistic, hands, holis, drawing = load_mediapipe_runtime()

cap = cv2.VideoCapture(0)

while True:
    lst = []
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for landmark in res.face_landmarks.landmark:
            lst.append(landmark.x - res.face_landmarks.landmark[1].x)
            lst.append(landmark.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for landmark in res.left_hand_landmarks.landmark:
                lst.append(landmark.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(landmark.y - res.left_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)

        if res.right_hand_landmarks:
            for landmark in res.right_hand_landmarks.landmark:
                lst.append(landmark.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(landmark.y - res.right_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)

        pred = label[np.argmax(model.predict(np.array(lst).reshape(1, -1), verbose=0))]
        cv2.putText(frm, str(pred), (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
