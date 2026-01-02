#!/usr/bin/env python3
"""
Setup script for Live Emotion Detection
This script helps you install dependencies and test your setup.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_model_files():
    """Check if model files exist"""
    print("🔍 Checking model files...")
    
    model_json = "model.json"
    model_h5 = "model.h5"
    
    if os.path.exists(model_json) and os.path.exists(model_h5):
        print("✅ Model files found!")
        return True
    else:
        print("❌ Model files not found!")
        print(f"   Looking for: {model_json} and {model_h5}")
        return False

def test_camera():
    """Test camera availability"""
    print("📹 Testing camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera is working!")
                return True
            else:
                print("❌ Camera opened but couldn't read frame")
                return False
        else:
            print("❌ Could not open camera")
            return False
    except ImportError:
        print("❌ OpenCV not installed")
        return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Live Emotion Detection Setup")
    print("=" * 40)
    
    # Check model files first
    if not check_model_files():
        print("\n💡 Make sure your trained model files (model.json and model.h5) are in the current directory")
        return
    
    # Install requirements
    if not install_requirements():
        print("\n💡 Try running: pip install -r requirements.txt")
        return
    
    # Test camera
    if not test_camera():
        print("\n💡 Check your camera connection and permissions")
        return
    
    print("\n🎉 Setup complete! You can now run:")
    print("   python live_emotion_detection.py")
    print("\n📝 Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 's' to save screenshot")

if __name__ == "__main__":
    main()
