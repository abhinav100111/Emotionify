from __future__ import annotations

import mediapipe as mp


def load_mediapipe_runtime():
    if not hasattr(mp, "solutions"):
        version = getattr(mp, "__version__", "unknown")
        raise RuntimeError(
            "Installed mediapipe version "
            f"{version} does not expose mp.solutions. "
            "Install the project requirements to get a compatible build: "
            "`pip install -r requirements.txt`."
        )

    holistic = mp.solutions.holistic
    hands = mp.solutions.hands
    holis = holistic.Holistic()
    drawing = mp.solutions.drawing_utils
    return holistic, hands, holis, drawing
