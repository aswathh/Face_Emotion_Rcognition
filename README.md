# Face Emotion Recognition

Detect faces in images and live webcam video, then classify each face's emotion (happy, sad, angry, surprise, and more) using **OpenCV** and **DeepFace**. Detected faces are outlined with a bounding box and labelled with the predicted emotion.

## Features

- **Image mode:** detect faces in a photo, draw a box around each one, and write the dominant emotion above it.
- **Video mode:** run the same pipeline frame by frame on a live webcam feed.
- **Face detection:** OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`).
- **Emotion classification:** DeepFace's emotion model (angry, disgust, fear, happy, sad, surprise, neutral).

## Project structure

```
Face_Emotion_Recognition/
├── FER_images.ipynb     # emotion recognition on still images
├── FER_video.ipynb      # emotion recognition on live webcam video
├── Images/              # sample images (create this folder yourself, see below)
├── pyproject.toml       # project dependencies
├── uv.lock              # locked dependency versions
├── .python-version      # Python version used by uv
└── README.md
```

## Tech stack

- Python 3.11
- [DeepFace](https://github.com/serengil/deepface) (TensorFlow backend)
- [OpenCV](https://opencv.org/)
- Matplotlib
- Jupyter notebooks in VS Code
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/aswathh/Face_Emotion_Recognition.git
cd Face_Emotion_Recognition
```

### 2. Install dependencies

```bash
uv sync
```

This creates a `.venv` folder and installs everything from `pyproject.toml`.

### 3. Add sample images

Create an `Images/` folder in the project root and add a few face photos, for example:

```
Images/
├── angry.jpg
├── happy.jpg
├── sad.jpg
└── surprise.jpg
```

### 4. Select the kernel

Open a notebook in VS Code and choose the project's `.venv` interpreter as the kernel (top right, **Select Kernel → Python Environments**).

## Usage

### Images: `FER_images.ipynb`

1. Load an image with `cv.imread("Images/happy.jpg")`.
2. Convert it to grayscale and detect faces with the Haar Cascade.
3. Analyze the emotion with DeepFace.
4. Draw the box and label, then display the result with Matplotlib.

```python
import cv2 as cv
from deepface import DeepFace

img = cv.imread("Images/happy.jpg")
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

for (x, y, w, h) in faces:
    face = img[y:y + h, x:x + w]
    result = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
    emotion = result[0]["dominant_emotion"]

    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText(img, emotion, (x, y - 10),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
```

### Video: `FER_video.ipynb`

Runs the same detection on each webcam frame. Press **q** in the video window to quit.

```python
import cv2 as cv
from deepface import DeepFace

video = cv.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=["emotion"])
        for face in result:
            r = face["region"]
            cv.rectangle(frame, (r["x"], r["y"]),
                         (r["x"] + r["w"], r["y"] + r["h"]), (0, 255, 0), 2)
            cv.putText(frame, face["dominant_emotion"], (r["x"], r["y"] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    except ValueError:
        pass  # no face detected in this frame

    cv.imshow("Emotion", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv.destroyAllWindows()
```

## Notes

- OpenCV loads images in **BGR** order, while Matplotlib expects **RGB**.
- The first DeepFace run downloads model weights, so it needs an internet connection and takes longer.
- Emotion predictions are estimates and can be wrong, especially with poor lighting, extreme angles, or partly hidden faces.

## Author

**aswathh** - [GitHub](https://github.com/aswathh)
