## 🧭 Overview

This system uses **landmark-based geometry** (not ML models) to detect ISL gestures using:

- ✋ Hand landmarks (21 points)
- 🧍 Body pose landmarks (33 points)

It converts gestures into:
- Words (e.g., *Namaste, Water, Yes*)
- Sentences (real-time)
- Optional speech output

---

## 🚀 What’s New in v2.0

- ✅ Chirality-aware thumb detection (Left vs Right)
- ✅ Uses 3D depth (z-axis)
- ✅ Angle-based finger detection
- ✅ Confidence scoring
- ✅ 18-frame smoothing window
- ✅ 65% stability threshold
- ✅ Gesture priority system
- ✅ Improved pinch & hook detection
- ✅ Better pose classification

---

## ✨ Features

- 🎯 Real-time gesture detection
- ✋ 20+ ISL hand signs
- 🧍 Body pose gestures
- 🧠 Smart smoothing (no flicker)
- 📊 Confidence percentage display
- 📝 Sentence builder
- 🔊 Text-to-Speech (optional)
- 🎮 Keyboard controls
- 📈 FPS monitoring

---

## 🛠️ Tech Stack

- Python 3.8+
- OpenCV
- MediaPipe (0.10.x)
- NumPy
- pyttsx3 (optional)

---

## ⚙️ How It Works

```

Webcam Input
↓
MediaPipe Detection (Hand + Pose)
↓
Geometric Analysis (angles, distances, depth)
↓
Gesture Classification
↓
Confidence + Smoothing
↓
Sentence Builder
↓
UI Display + Optional Speech

````

---

## 🤟 Supported Gestures

### ✋ Hand Gestures

- THUMBS UP → Yes / Good  
- THUMBS DOWN → No / Bad  
- NAMASTE → Hello  
- I LOVE YOU  
- OK  
- CALL ME  
- ONE / TWO / THREE / FOUR / FIVE  
- OPEN HAND → Stop  
- EAT / FOOD  
- WATER  
- SCISSORS  
- POWER  
- FIST  

---

### 🧍 Body Gestures

- CELEBRATE  
- HANDS UP  
- WELCOME  
- WAVE  
- PRAY  
- CONFUSED  
- HUG  

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/isl-recognition.git
cd isl-recognition
````

---

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install opencv-python mediapipe numpy
```

Optional (for speech):

```bash
pip install pyttsx3
```

---

### 4. First Run Note

* Automatically downloads:

  * `hand_landmarker.task`
  * `pose_landmarker.task`
* Size: ~20MB
* Requires internet (first run only)

---

## ▶️ How to Run

```bash
python main.py
```

---

## 🎮 Controls

| Key        | Action         |
| ---------- | -------------- |
| Hold 2 sec | Auto-save word |
| SPACE      | Save instantly |
| C          | Clear sentence |
| S          | Speak sentence |
| ESC        | Exit           |

---

## 📁 Project Structure

```
📦 isl-recognition/
 ┣ 📄 main.py
 ┣ 📄 README.md
 ┣ 📄 hand_landmarker.task
 ┗ 📄 pose_landmarker.task
```

---

## ⚠️ Troubleshooting

### Webcam not opening

Change:

```python
cv2.VideoCapture(0)
```

To:

```python
cv2.VideoCapture(1)
```

---

### Low accuracy

* Improve lighting
* Keep hand fully visible
* Avoid fast movement

---

### TTS not working

```bash
pip install pyttsx3
```

---

## 🔮 Future Improvements

* ML-based gesture recognition (CNN)
* Full ISL alphabet (A–Z)
* Web app (Streamlit/Flask)
* Mobile version
* Two-hand gestures
* Regional ISL support

---

## 🎯 Applications

* Assistive tech for hearing-impaired
* ISL learning tool
* Gesture-based interfaces
* Smart communication systems

---

## 👨‍💻 Author

**Prerna Prachi**
B.Tech CSE (AIML)
VIT Bhopal University

---

## 📄 License

Educational use only.
All processing is done locally (no data sharing).

---

## ⭐ Support

If you like this project:

* ⭐ Star the repo
* 🍴 Fork it
* 🛠 Improve it

```
