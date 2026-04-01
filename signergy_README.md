# 🤟 Signergy — Real-Time Sign Language Recognition

A real-time sign language recognition system that uses your webcam to detect hand gestures and translate them into text. Built with **MediaPipe**, **OpenCV**, and a **Random Forest** classifier trained on landmark coordinates.

---

## 🎥 Demo

> Point your hand at the webcam — Signergy detects your hand landmarks, classifies the gesture, and builds up a text output in real time.

---

## 🧠 How It Works

1. **Hand Detection** — MediaPipe detects 21 hand landmarks (x, y, z) per frame
2. **Feature Extraction** — Landmark coordinates + wrist-to-fingertip distances are extracted as a feature vector
3. **Classification** — A trained Random Forest model predicts the sign from the feature vector
4. **UI** — OpenCV renders the live camera feed with an overlay panel showing the detected text

---

## 🗂️ Project Structure

```
├── signergy.py          # Main application (all classes and logic)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

### Classes

| Class | Responsibility |
|---|---|
| `HandTracker` | Wraps MediaPipe Hands — detects landmarks and extracts features |
| `SignLanguageRecognizer` | Trains, saves, loads, and runs the Random Forest model |
| `SignLanguageUI` | OpenCV real-time UI with camera feed and text panel |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

The model trains on a CSV dataset of hand landmark coordinates with a label column.  
A compatible dataset can be found on Kaggle (e.g. ASL/BSL landmark datasets).

```bash
# Optional: process a raw Kaggle dataset first
python signergy.py --process-dataset path/to/raw_dataset.csv
```

### 3. Train the model

```bash
python signergy.py --train --dataset path/to/dataset.csv --model sign_language_model.pkl
```

### 4. Run the live recognition app

```bash
python signergy.py --model sign_language_model.pkl
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| `Q` | Quit the application |
| `C` | Clear the text buffer |
| `S` | Save the current text to `sign_language_text.txt` |

---

## ⚙️ Configuration

You can adjust these settings inside `signergy.py`:

| Parameter | Default | Description |
|---|---|---|
| `detection_confidence` | `0.5` | MediaPipe hand detection threshold |
| `tracking_confidence` | `0.5` | MediaPipe hand tracking threshold |
| `detection_cooldown` | `1.0s` | Min time between sign detections |
| `confidence_threshold` | `0.7` | Minimum model confidence to accept a prediction |
| `n_estimators` | `100` | Number of trees in the Random Forest |

---

## 📦 Requirements

- Python 3.8+
- Webcam

```
opencv-python
mediapipe
numpy
pandas
scikit-learn
```

---

## 🔭 Future Improvements

- [ ] Add support for two-handed signs
- [ ] Integrate a sequence model (LSTM) for dynamic gestures
- [ ] Add confidence score display in the UI
- [ ] Support for multiple sign languages
- [ ] Export recognised text to speech (TTS)

---
