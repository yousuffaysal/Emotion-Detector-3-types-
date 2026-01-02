
# End-to-End Emotion Detection Project 🎭

A state-of-the-art, real-time emotion detection web application. This project uses deep learning to analyze facial expressions from a webcam feed and display the detected emotion (Happy 😊, Sad 😢, Neutral 😐) in a premium, responsive user interface.

## 🛠️ Tech Stack Used

This project is built using a modern **full-stack** architecture:

**Backend (AI & API)**
*   **Language**: Python 3.9+ 🐍
*   **Framework**: Flask (REST API)
*   **Computer Vision**: OpenCV (Face detection, image processing)
*   **Deep Learning**: TensorFlow/Keras (CNN Model inference)
*   **Numerical Processing**: NumPy

**Frontend (User Interface)**
*   **Framework**: React (Vite) ⚛️
*   **Styling**: Tailwind CSS + Custom CSS (Ivory Theme, premium typography)
*   **Animation**: Framer Motion (Smooth transitions)
*   **Fonts**: Italiana (Headers), Manrope (Body)

---

## ✨ Features

*   **Real-time AI Analysis**: Instant emotion classification from video stream.
*   **Premium UI/UX**: "Clean, Minimal, Curated Gallery" aesthetic using an Ivory (`#fffff0`) theme.
*   **Smart Detection**: Automatically crops faces and provides confident predictions.
*   **Model Selection**: Architecture supports loading multiple `.h5` models (currently using a robust FER2013-trained model).
*   **Responsive**: Works seamlessly on desktop and mobile browsers.

---

## 🚀 How to Run (End-to-End)

To run the full application, you need to start **both** the backend (Python) and the frontend (React) servers. Open **two separate terminals**.

### Terminal 1: Backend Server

1.  Navigate to the `facedetect` directory:
    ```bash
    cd /path/to/facedetect
    ```
2.  Install python dependencies (if not done):
    ```bash
    pip install -r requirements.txt
    ```
3.  Start the Flask server:
    ```bash
    python backend/app.py
    ```
    *The backend will start at `http://localhost:5001`*

### Terminal 2: Frontend Client

1.  Navigate to the `frontend` directory:
    ```bash
    cd /path/to/facedetect/frontend
    ```
2.  Install node dependencies (first time only):
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
    *The frontend will start at `http://localhost:5173`*

---

## 🎯 Usage

1.  Open your browser and visit **`http://localhost:5173`**.
2.  Allow camera permissions when prompted.
3.   The AI will immediately start analyzing your face.
    *   **Searching...**: No face detected.
    *   **Happy/Sad/Neutral**: Detected emotion with confidence score.
4.  Use the dropdown menu to simulate model switching (loads architecture).

## 📁 Project Structure

```
facedetect/
├── backend/            # Python Flask API
│   ├── app.py          # Server logic & Inference
│   └── model.h5        # Deep Learning Model Weights
├── frontend/           # React Application
│   ├── src/            # Components & Styles
│   └── index.html      # Entry point
├── model.json          # Model Architecture Definition
└── README.md           # Documentation
```
