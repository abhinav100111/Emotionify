from pathlib import Path
import os
import webbrowser

os.environ.setdefault("KERAS_BACKEND", "jax")

import av
import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer

from emotionify_runtime import load_mediapipe_runtime

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
LABELS_PATH = BASE_DIR / "labels.npy"
EMOTION_PATH = BASE_DIR / "emotion.npy"

model = load_model(MODEL_PATH)
label = np.load(LABELS_PATH, allow_pickle=True)
holistic, hands, holis, drawing = load_mediapipe_runtime()

st.set_page_config(page_title="Emotionify", page_icon=":performing_arts:", layout="centered")
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #6c63ff;
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .stButton>button {
        color: white;
        background-color: #6c63ff;
        border-radius: 10px;
        height: 50px;
        font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("Emotionify - Music and Movie Recommender")

if "run" not in st.session_state:
    st.session_state["run"] = "true"


class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

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
            np.save(EMOTION_PATH, np.array([pred]))

        drawing.draw_landmarks(
            frm,
            res.face_landmarks,
            holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
            connection_drawing_spec=drawing.DrawingSpec(thickness=1),
        )
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


with st.expander("Customize Your Preferences", expanded=True):
    lang = st.text_input("Enter Language", placeholder="e.g., English, Hindi, Spanish...")
    singer = st.text_input("Favorite Singer (Optional)", placeholder="e.g., Arijit Singh, Taylor Swift...")
    choice = st.selectbox("What would you like recommended?", ["Songs", "Movies"])

if lang and st.session_state["run"] != "false":
    st.markdown("### Capturing your emotions...")
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

btn = st.button("Recommend Me")

if btn:
    try:
        emotion = np.load(EMOTION_PATH, allow_pickle=True)[0]
    except Exception:
        emotion = ""

    if not emotion:
        st.warning("Please capture your emotion first.")
        st.session_state["run"] = "true"
    else:
        emotion_lower = str(emotion).lower()

        if choice == "Songs":
            if emotion_lower == "sad":
                search_query = f"{lang} happy uplifting songs {singer}"
            elif emotion_lower == "fear":
                search_query = f"{lang} funny cheerful songs {singer}"
            elif emotion_lower == "angry":
                search_query = f"{lang} calm relaxing peaceful songs {singer}"
            else:
                search_query = f"{lang} {emotion} song {singer}"
        else:
            if emotion_lower == "sad":
                search_query = f"{lang} happy feel-good comedy movie"
            elif emotion_lower == "fear":
                search_query = f"{lang} funny light-hearted comedy movie"
            elif emotion_lower == "angry":
                search_query = f"{lang} calm relaxing peaceful movie"
            else:
                search_query = f"{lang} {emotion} mood movie"

        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
        np.save(EMOTION_PATH, np.array([""]))
        st.session_state["run"] = "false"
