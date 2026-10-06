# Real-Time Facial Skin Condition Detection

A real-time **Computer Vision system** that detects visible facial skin conditions such as **Acne, Dark Circles, and Wrinkles** from live webcam input using a custom-trained YOLO object detection model.

The model was trained on approximately **9,000 facial images** and integrated with real-time webcam inference to demonstrate live detection.

---

## 🚀 Project Overview

The goal of this project is to build a computer vision application capable of identifying common visible facial features/skin conditions from real-time camera input.

The system takes webcam frames as input, runs them through a trained YOLO model, and displays detected conditions with bounding boxes and confidence scores.

### Detects

* **Acne**
* **Dark Circles**
* **Wrinkles**

---

## 🧠 How It Works

```text
Webcam
   ↓
Live Video Frames
   ↓
Image Preprocessing
   ↓
YOLO Object Detection Model
   ↓
Detection + Confidence Score
   ↓
Bounding Boxes
   ↓
Real-Time Display
```

---

## 📊 Dataset

The model was trained using approximately:

**9,000 facial images**

The dataset contains annotated examples of:

* Acne
* Dark Circles
* Wrinkles

The dataset was prepared for object detection so that the model learns both **what condition is present** and **where it appears in the image**.

---

## 🤖 Model

The project uses a **YOLO-based object detection architecture** trained specifically for the three target classes.

### Classes

```text
0 → Acnes
1 → Dark Circle
2 → Wrinkle
```

The trained model produces:

* Detected class
* Bounding box coordinates
* Confidence score

Example:

```text
Dark Circle   0.87
Acnes         0.81
Wrinkle       0.76
```

---

## 💻 Real-Time Detection

The trained model is connected to a webcam pipeline.

For every incoming frame:

1. Capture frame from webcam
2. Pass frame to the YOLO model
3. Detect facial conditions
4. Draw bounding boxes
5. Display class names and confidence scores
6. Continue inference on the next frame

This allows the model to perform detection continuously on live video.

---

## 🛠️ Tech Stack

* **Python**
* **YOLO**
* **Ultralytics**
* **OpenCV**
* **Roboflow**
* **Google Colab**
* **Computer Vision**
* **Object Detection**

---

## 📁 Project Structure

```text
Real-Time-Facial-Skin-Detection/
│
├── training/
│   └── model_training.ipynb
│
├── inference/
│   └── realtime_detection.py
│
├── weights/
│   └── best.pt
│
├── README.md
└── requirements.txt
```

---

## 🔬 Model Training

The model was trained using an annotated facial-image dataset.

The training workflow included:

```text
Dataset Collection
       ↓
Image Annotation
       ↓
Dataset Preparation
       ↓
YOLO Training
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Real-Time Inference
```

The best-performing model weights were saved as:

```text
best.pt
```

---

## 📸 Example Output

The real-time application displays detections directly on the webcam feed.

Example output:

```text
┌─────────────────────────────┐
│                             │
│     [Dark Circle 0.89]      │
│          👁                 │
│                             │
│       [Acnes 0.82]          │
│                             │
│      [Wrinkle 0.76]         │
│                             │
└─────────────────────────────┘
```

---

## 🎯 Key Learning Outcomes

Through this project, I worked on:

* Computer vision model training
* Object detection
* Image annotation and dataset preparation
* YOLO model training
* Model evaluation
* OpenCV-based image processing
* Real-time webcam inference
* Deploying trained models for live prediction

---

## ⚠️ Disclaimer

This project is intended for **computer vision experimentation and educational purposes**.

The detected categories represent visual patterns learned from the training dataset and should **not be considered medical diagnoses or professional dermatological assessments**.

---

## 🔮 Future Improvements

Potential improvements include:

* Face detection and facial-region tracking
* Skin-zone analysis for forehead, cheeks, and chin
* Improved detection accuracy with a larger and more diverse dataset
* Model optimization for faster real-time inference
* Tracking detected conditions across video frames
* Building a simple web interface using Streamlit
* Adding historical results for comparison over time

---

## 👨‍💻 Author

**Ansh Santosh Kambli**

Data Scientist | Data Analyst

GitHub: `https://github.com/AnshKambli`

LinkedIn: `https://www.linkedin.com/in/ansh-kambli-3598b1251/`
