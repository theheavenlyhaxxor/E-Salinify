import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- SETTINGS ---
video_input = '3.mp4'
model_asset_path = 'hand_landmarker.task'  # Path to downloaded mediapipe model
smnist_model = load_model('smnist.h5')     # Your existing A-Z model
CONFIDENCE_THRESHOLD = 0.85
STABILITY_FRAMES = 10

# Labels
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# --- MEDIAPIPE INITIALIZATION ---
# Using the modern Tasks API instead of the deprecated solutions.hands
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_asset_path),
    running_mode=VisionRunningMode.VIDEO,  # Optimized for sequential frames
    num_hands=1
)

# --- STATE VARIABLES ---
cap = cv2.VideoCapture(video_input)
w, h = int(cap.get(3)), int(cap.get(4))
translated_text = ""
current_prediction_list = []
last_confirmed_letter = ""
y_pred_all = []

# Create the landmarker instance
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Calculate timestamp for VIDEO mode (required for modern MediaPipe)
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Convert frame to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform Detection
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        if result.hand_landmarks:
            # Process the first hand detected
            landmarks = result.hand_landmarks[0]
            
            # Calculate Bounding Box
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            x_min, x_max = int(min(x_coords) * w) - 25, int(max(x_coords) * w) + 25
            y_min, y_max = int(min(y_coords) * h) - 25, int(max(y_coords) * h) + 25
            
            # Clipping to frame boundaries
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            try:
                # Pre-processing (Crop -> Gray -> Resize -> Reshape)
                roi = cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(roi, (28, 28))
                pixeldata = roi.reshape(1, 28, 28, 1) / 255.0

                # Prediction
                prediction = smnist_model.predict(pixeldata, verbose=0)
                idx = np.argmax(prediction[0])
                confidence = prediction[0][idx]
                predicted_char = letterpred[idx]
                
                y_pred_all.append(idx)

                # Stability Filter for accuracy
                if confidence > CONFIDENCE_THRESHOLD:
                    current_prediction_list.append(predicted_char)
                    if len(current_prediction_list) > STABILITY_FRAMES:
                        current_prediction_list.pop(0)

                    if len(current_prediction_list) == STABILITY_FRAMES and len(set(current_prediction_list)) == 1:
                        confirmed_letter = current_prediction_list[0]
                        if confirmed_letter != last_confirmed_letter:
                            translated_text += confirmed_letter
                            last_confirmed_letter = confirmed_letter

                # Draw UI
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{predicted_char} {confidence*100:.0f}%", (x_min, y_min-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception:
                continue

        # Display result
        cv2.putText(frame, f"TEXT: {translated_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Sign Translator", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
print(f"Final Translated Output: {translated_text}")