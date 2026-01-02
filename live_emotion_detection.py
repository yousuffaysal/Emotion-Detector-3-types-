#!/usr/bin/env python3
"""
Live Emotion Detection using Webcam
This script uses your trained emotion recognition model to detect emotions in real-time from webcam feed.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import time

class LiveEmotionDetector:
    def __init__(self, model_json_path, model_weights_path):
        """
        Initialize the emotion detector with the trained model
        
        Args:
            model_json_path (str): Path to the model architecture JSON file
            model_weights_path (str): Path to the model weights H5 file
        """
        # Load the model architecture
        with open(model_json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        
        # Create model from JSON
        self.model = model_from_json(loaded_model_json)
        
        # Load weights
        self.model.load_weights(model_weights_path)
        
        # Compile the model (required for prediction)
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Emotion mapping based on your model (3 classes: happy, sad, neutral)
        self.emotion_labels = ["😊 Happy", "😢 Sad", "😐 Neutral"]
        
        # Initialize face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Statistics tracking
        self.total_detections = 0
        self.emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
        self.session_start_time = time.time()
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.dominant_emotion = "N/A"
        self.dominant_confidence = 0.0
        
        print("✅ Model loaded successfully!")
        print(f"📊 Detecting {len(self.emotion_labels)} emotions: {', '.join(self.emotion_labels)}")
    
    def preprocess_face(self, face_roi):
        """
        Preprocess face region for emotion prediction
        
        Args:
            face_roi (numpy.ndarray): Face region from camera frame
            
        Returns:
            numpy.ndarray: Preprocessed face ready for model prediction
        """
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize to 48x48 (model input size)
        face_resized = cv2.resize(face_roi, (48, 48))
        
        # Normalize pixel values to [0, 1]
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Reshape for model input (batch_size, height, width, channels)
        face_reshaped = face_normalized.reshape(1, 48, 48, 1)
        
        return face_reshaped
    
    def predict_emotion(self, face_roi):
        """
        Predict emotion from face region
        
        Args:
            face_roi (numpy.ndarray): Face region from camera frame
            
        Returns:
            tuple: (emotion_label, confidence_score)
        """
        # Preprocess the face
        processed_face = self.preprocess_face(face_roi)
        
        # Make prediction
        predictions = self.model.predict(processed_face, verbose=0)
        
        # Get emotion with highest confidence
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        
        emotion_label = self.emotion_labels[emotion_idx]
        
        # Update statistics
        self.total_detections += 1
        self.emotion_counts[emotion_label] += 1
        
        # Update dominant emotion
        if confidence > self.dominant_confidence:
            self.dominant_emotion = emotion_label
            self.dominant_confidence = confidence
        
        return emotion_label, confidence
    
    def draw_emotion_info(self, frame, emotion, confidence, x, y, w, h):
        """
        Draw emotion information on the frame
        
        Args:
            frame (numpy.ndarray): Camera frame
            emotion (str): Detected emotion
            confidence (float): Confidence score
            x, y, w, h (int): Face bounding box coordinates
        """
        # Sidebar width
        sidebar_width = 280
        frame_width = frame.shape[1]
        sidebar_x = frame_width - sidebar_width
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Prepare text
        emotion_text = f"{emotion}"
        confidence_text = f"Confidence: {confidence:.2f}"
        location_text = f"Location: ({x},{y}) Size: {w}x{h}"
        
        # Adjust text position if face is near the sidebar area
        # If face is too close to sidebar, draw info above or to the left
        text_x = x
        if x + w > sidebar_x - 150:  # If face overlaps sidebar area
            # Draw info above face instead of beside it
            text_x = max(10, x - 200) if x > 200 else x
        
        # Draw emotion label
        cv2.putText(frame, emotion_text, (text_x, max(25, y-10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw confidence score
        cv2.putText(frame, confidence_text, (text_x, y+h+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw location info (only if not near sidebar)
        if x + w < sidebar_x - 50:
            cv2.putText(frame, location_text, (text_x, y+h+45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_statistics_overlay(self, frame):
        """
        Draw comprehensive statistics overlay on the right side of the frame
        
        Args:
            frame (numpy.ndarray): Camera frame
        """
        # Calculate session time
        session_time = time.time() - self.session_start_time
        minutes = int(session_time // 60)
        seconds = int(session_time % 60)
        
        # Get device name (simplified)
        import os
        device_name = os.environ.get('USER', 'Unknown') + "'s Device"
        
        # Sidebar width - adjust to be narrower
        sidebar_width = 280
        frame_height, frame_width = frame.shape[:2]
        
        # Start position of sidebar (right side)
        sidebar_x = frame_width - sidebar_width
        
        # Create semi-transparent sidebar background
        overlay = frame[:, sidebar_x:].copy()
        cv2.rectangle(overlay, (0, 0), (sidebar_width, frame_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame[:, sidebar_x:], 0.25, 0, frame[:, sidebar_x:])
        
        # Add border line separating sidebar from main view
        cv2.line(frame, (sidebar_x, 0), (sidebar_x, frame_height), (255, 255, 255), 2)
        
        # Starting Y position
        start_y = 20
        line_spacing = 30
        
        # Title
        cv2.putText(frame, "Emotion Live", (sidebar_x + 10, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 182, 193), 2)
        
        start_y += line_spacing + 5
        
        # Device info
        cv2.putText(frame, f"Device:", (sidebar_x + 10, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # Split long device name into multiple lines if needed
        device_display = device_name if len(device_name) < 25 else device_name[:22] + "..."
        cv2.putText(frame, device_display, (sidebar_x + 10, start_y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        start_y += line_spacing + 10
        
        # Detection count
        cv2.putText(frame, f"Detections: {self.total_detections}", (sidebar_x + 10, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        start_y += line_spacing
        
        # Session time
        cv2.putText(frame, f"Session: {minutes:02d}m {seconds:02d}s", (sidebar_x + 10, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        start_y += line_spacing
        
        # FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (sidebar_x + 10, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        start_y += line_spacing + 5
        
        # Separator line
        cv2.line(frame, (sidebar_x + 10, start_y), (frame_width - 10, start_y), (100, 100, 100), 1)
        
        start_y += line_spacing
        
        # Dominant emotion
        dominant_color = (0, 255, 0) if self.dominant_emotion != "N/A" else (255, 255, 255)
        cv2.putText(frame, "Dominant:", (sidebar_x + 10, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"{self.dominant_emotion}", (sidebar_x + 10, start_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, dominant_color, 2)
        
        start_y += line_spacing + 15
        
        # Confidence
        cv2.putText(frame, f"Confidence: {self.dominant_confidence:.1%}", (sidebar_x + 10, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        start_y += line_spacing + 10
        
        # Separator line
        cv2.line(frame, (sidebar_x + 10, start_y), (frame_width - 10, start_y), (100, 100, 100), 1)
        
        start_y += line_spacing
        
        # Emotion breakdown header
        cv2.putText(frame, "Emotion Stats:", (sidebar_x + 10, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        start_y += line_spacing
        
        # Emotion breakdown items
        for emotion, count in self.emotion_counts.items():
            percentage = (count / max(self.total_detections, 1)) * 100
            # Use different colors for each emotion
            if "Happy" in emotion:
                color = (0, 255, 100)
            elif "Sad" in emotion:
                color = (100, 100, 255)
            else:
                color = (255, 255, 100)
            
            cv2.putText(frame, f"{emotion}:", (sidebar_x + 10, start_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(frame, f"{count} ({percentage:.1f}%)", (sidebar_x + 10, start_y + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Progress bar
            bar_width = int(percentage * 2.2)  # Scale for sidebar
            bar_height = 6
            cv2.rectangle(frame, (sidebar_x + 10, start_y + 25), 
                         (sidebar_x + 10 + bar_width, start_y + 25 + bar_height), color, -1)
            cv2.rectangle(frame, (sidebar_x + 10, start_y + 25), 
                         (sidebar_x + 10 + 240, start_y + 25 + bar_height), (100, 100, 100), 1)
            
            start_y += 50
    
    def run_live_detection(self, camera_index=0):
        """
        Run live emotion detection from webcam
        
        Args:
            camera_index (int): Camera device index (usually 0 for default camera)
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera!")
            return
        
        print("🎥 Starting live emotion detection...")
        print("📝 Press 'q' to quit, 's' to save screenshot, 'r' to reset stats")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Track consecutive errors
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        try:
            while True:
                try:
                    ret, frame = cap.read()
                    
                    # If frame read fails, retry instead of breaking
                    if not ret:
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print("❌ Error: Too many consecutive frame read failures!")
                            print("💡 Trying to reinitialize camera...")
                            cap.release()
                            time.sleep(0.5)
                            cap = cv2.VideoCapture(camera_index)
                            if cap.isOpened():
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                consecutive_errors = 0
                                print("✅ Camera reinitialized successfully!")
                            else:
                                print("❌ Failed to reinitialize camera!")
                                break
                        else:
                            time.sleep(0.01)  # Short delay before retry
                            continue
                    
                    # Reset error counter on successful read
                    consecutive_errors = 0
                    
                    # Check if window was closed by user (only works if window was created)
                    try:
                        window_prop = cv2.getWindowProperty('Live Emotion Detection', cv2.WND_PROP_VISIBLE)
                        if window_prop < 1:
                            print("🪟 Window closed by user")
                            break
                    except:
                        pass  # Window check might fail, continue anyway
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    faces = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, 
                        minNeighbors=5, 
                        minSize=(30, 30)
                    )
                    
                    # Process each detected face
                    for (x, y, w, h) in faces:
                        try:
                            # Extract face region
                            face_roi = gray[y:y+h, x:x+w]
                            
                            # Skip if face region is too small or invalid
                            if face_roi.size == 0 or w < 30 or h < 30:
                                continue
                            
                            # Predict emotion
                            emotion, confidence = self.predict_emotion(face_roi)
                            
                            # Draw information on frame
                            self.draw_emotion_info(frame, emotion, confidence, x, y, w, h)
                        except Exception as e:
                            # Continue processing other faces even if one fails
                            print(f"⚠️ Warning: Error processing face - {e}")
                            continue
                    
                    # Update FPS calculation
                    self.frame_count += 1
                    if self.frame_count % 30 == 0:  # Update FPS every 30 frames
                        fps_end_time = time.time()
                        self.current_fps = 30 / (fps_end_time - self.fps_start_time)
                        self.fps_start_time = fps_end_time
                    
                    # Draw comprehensive statistics overlay (on right side)
                    self.draw_statistics_overlay(frame)
                    
                    # Add instructions at bottom left (away from sidebar)
                    instruction_text = "Press 'q' to quit, 's' to save, 'r' to reset"
                    # Position at bottom left, but not under the sidebar
                    sidebar_width = 280
                    cv2.putText(frame, instruction_text, 
                               (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Display frame
                    cv2.imshow('Live Emotion Detection', frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("👋 Quitting...")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"emotion_screenshot_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"📸 Screenshot saved as {filename}")
                    elif key == ord('r'):
                        # Reset statistics
                        self.total_detections = 0
                        self.emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
                        self.dominant_emotion = "N/A"
                        self.dominant_confidence = 0.0
                        print("🔄 Statistics reset!")
                
                except cv2.error as e:
                    print(f"⚠️ OpenCV error: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print("❌ Too many OpenCV errors, exiting...")
                        break
                    time.sleep(0.1)
                    continue
                
                except Exception as e:
                    print(f"⚠️ Unexpected error: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print("❌ Too many errors, exiting...")
                        break
                    time.sleep(0.1)
                    continue
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        
        except Exception as e:
            print(f"❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            try:
                cap.release()
                cv2.destroyAllWindows()
                print("✅ Camera released and windows closed")
            except:
                pass  # Ignore errors during cleanup

def main():
    """Main function to run the live emotion detection"""
    
    # Model file paths
    model_json_path = "model.json"
    model_weights_path = "model.h5"
    
    print("🎭 Live Emotion Detection")
    print("=" * 40)
    
    try:
        # Check if model files exist
        import os
        if not os.path.exists(model_json_path):
            print(f"❌ Error: {model_json_path} not found!")
            return
        if not os.path.exists(model_weights_path):
            print(f"❌ Error: {model_weights_path} not found!")
            return
        
        print("✅ Model files found")
        
        # Initialize emotion detector
        detector = LiveEmotionDetector(model_json_path, model_weights_path)
        
        # Run live detection
        detector.run_live_detection(camera_index=0)
        
    except FileNotFoundError as e:
        print(f"❌ Error: Model file not found - {e}")
        print("📁 Make sure 'model.json' and 'model.h5' are in the current directory")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Please check your camera connection and try again")
        print("💡 Try running: python setup.py")

if __name__ == "__main__":
    main()
