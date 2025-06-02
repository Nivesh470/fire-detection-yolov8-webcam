# 🔥 Real-Time Fire Detection using Webcam with YOLOv8 and HSV Masking

This project combines deep learning with classical computer vision to detect fire using a laptop webcam. It uses a pretrained YOLOv8 model (`best.pt`) and HSV color segmentation to identify potential flame regions and alert the user in real-time.

---

## 📌 Features

- 🔍 Detects fire in real-time using webcam
- 🧠 YOLOv8 deep learning model for accurate detection
- 🎨 HSV color range detection for flame-colored regions
- 🔄 Flicker-based validation to reduce false positives from static lights
- 👨‍💻 Python-based; easily extendable to hardware (like Arduino/ESP32)

---

## 📽️ Demo

*Live webcam demo output:*  
[Include screenshot or demo GIF here]

---

## 🛠️ Requirements

- Python 3.8+
- Webcam (built-in or external)
- PyCharm (or any Python IDE)
- Packages:
  - `opencv-python`
  - `numpy`
  - `ultralytics`

### 🔧 Install dependencies:

```bash
pip install opencv-python numpy ultralytics
