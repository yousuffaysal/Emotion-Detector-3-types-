# Live Emotion Detection 🎭

A real-time emotion detection application that uses your trained deep learning model to detect emotions from live camera feed.

## Features ✨

- **Real-time emotion detection** from webcam feed
- **3 emotion classes**: Happy 😊, Sad 😢, Neutral 😐
- **Face detection** using OpenCV Haar cascades
- **Confidence scores** for each prediction
- **FPS counter** for performance monitoring
- **Screenshot capture** functionality
- **Mirror mode** for natural interaction

## Quick Start 🚀

### 1. Install Dependencies
```bash
# Run the setup script (recommended)
python setup.py

# Or install manually
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python live_emotion_detection.py
```

### 3. Controls
- **'q'** - Quit the application
- **'s'** - Save a screenshot

## Requirements 📋

- Python 3.7+
- Webcam/Camera
- Your trained model files:
  - `model.json` (model architecture)
  - `model.h5` (model weights)

## Model Information 🧠

Your model is trained on the FER2013 dataset and detects 3 emotions:
- **Happy** (😊)
- **Sad** (😢) 
- **Neutral** (😐)

The model expects:
- Input size: 48x48 grayscale images
- Normalized pixel values (0-1 range)
- Single channel (grayscale)

## Troubleshooting 🔧

### Camera Issues
- Make sure your camera is connected and not being used by other applications
- Check camera permissions in your system settings
- Try different camera indices (0, 1, 2) if you have multiple cameras

### Model Issues
- Ensure `model.json` and `model.h5` files are in the same directory as the script
- Check that the model files are not corrupted

### Performance Issues
- Close other applications using the camera
- Reduce camera resolution if needed
- The application automatically adjusts FPS for optimal performance

## File Structure 📁

```
facedetect/
├── live_emotion_detection.py  # Main application
├── setup.py                   # Setup and testing script
├── requirements.txt           # Python dependencies
├── model.json                 # Model architecture
├── model.h5                   # Model weights
└── README.md                  # This file
```

## Technical Details 🔬

- **Face Detection**: OpenCV Haar Cascade Classifier
- **Preprocessing**: Grayscale conversion, resizing to 48x48, normalization
- **Model**: Deep Convolutional Neural Network (DCNN)
- **Framework**: TensorFlow/Keras
- **Performance**: Optimized for real-time processing

## Customization 🛠️

You can modify the application by:
- Changing emotion labels in the `emotion_labels` list
- Adjusting camera resolution in the `run_live_detection` method
- Modifying face detection parameters for better accuracy
- Adding new features like emotion history tracking

## Support 💬

If you encounter any issues:
1. Run `python setup.py` to check your setup
2. Verify all dependencies are installed
3. Check camera permissions and availability
4. Ensure model files are present and valid

Enjoy detecting emotions in real-time! 🎉
