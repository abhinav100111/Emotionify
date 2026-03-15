# Emotionify

Emotionify is a webcam-based emotion detection project built with Streamlit, MediaPipe, OpenCV, and Keras. It captures face and hand landmarks in real time, predicts an emotion from a trained model, and then opens recommendation results based on the detected mood.

## What this project does

- Detects emotion from live webcam input
- Uses face and hand landmarks extracted with MediaPipe
- Loads a trained Keras model from `model.h5`
- Recommends songs or movies based on mood, language, and optional singer preference
- Opens the recommendation search in YouTube

## Project structure

Here are the main files in this repository:

- `music.py` - main Streamlit app for live emotion capture and recommendations
- `emotionify_runtime.py` - MediaPipe compatibility loader used by the app and data scripts
- `data_collection.py` - collects landmark samples from webcam and saves them as `.npy`
- `data_training.py` - trains the emotion classification model and saves `model.h5` and `labels.npy`
- `model.h5` - trained emotion detection model
- `labels.npy` - label mapping used by the trained model
- `emotion.npy` - temporary file used to store the latest detected emotion
- `setup_project.bat` - Windows setup script for creating the virtual environment and installing dependencies
- `run_project.bat` - Windows launcher for the Streamlit app
- `requirements.txt` - Python dependencies
- `FER+_ResNet_VGG_Integration.ipynb` - notebook included with the project

## Recommended way to run

This project is currently set up most cleanly for Windows with Python 3.12.

1. Run the setup script:

```bat
setup_project.bat
```

2. Start the app:

```bat
run_project.bat
```

3. Open the Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Manual run

If you prefer to run it yourself:

```bat
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run music.py
```

## How to use the app

1. Enter your preferred language.
2. Optionally enter a singer name.
3. Choose whether you want song or movie recommendations.
4. Allow webcam access.
5. Let the app capture your emotion.
6. Click `Recommend Me`.

The app will use the detected mood to build a recommendation query and open the result in your browser.

## Training your own model

If you want to retrain the project on your own samples:

1. Run `data_collection.py` and enter an emotion name when prompted.
2. Repeat for each emotion you want to collect.
3. Run `data_training.py` to generate a fresh `model.h5` and `labels.npy`.

Example:

```bat
.\.venv\Scripts\python.exe data_collection.py
.\.venv\Scripts\python.exe data_training.py
```

## Dependencies

Main libraries used in this project:

- Streamlit
- streamlit-webrtc
- Keras
- OpenCV
- NumPy
- Pillow
- MediaPipe
- av

## Notes before pushing to GitHub

- `model.h5` is already included, so the repository contains a large binary file.
- `.venv` and `__pycache__` are local environment folders and usually should not be committed.
- `emotion.npy` is generated during use and does not need to be versioned unless you explicitly want it in the repo.
- `app.py` appears to be an older Flask-based experiment and is not the main runnable app for this repository.

## Screenshots

The repository also includes sample images:

- `frontend1.jpg`
- `frontend2.jpg`
- `frontend3.jpg`
- `frontend4.jpg`
