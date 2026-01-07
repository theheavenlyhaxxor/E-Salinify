import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd

# 1. Load your model
model = load_model('smnist.h5')

mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 2. CHANGE THIS: Put the path to your recorded video file here
video_path = '4.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties for scaling
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

translated_text = ""
last_predicted_letter = ""
frames_consistent = 0  # To ensure we don't spam the same letter

print("Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # End of video

    # Convert to RGB for MediaPipe
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    
    if result.multi_hand_landmarks:
        for handLMs in result.multi_hand_landmarks:
            # Calculate Bounding Box
            x_max, y_max = 0, 0
            x_min, y_min = w, h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max: x_max = x
                if x < x_min: x_min = x
                if y > y_max: y_max = y
                if y < y_min: y_min = y
            
            # Add padding and ensure within frame boundaries
            y_min = max(0, y_min - 20)
            y_max = min(h, y_max + 20)
            x_min = max(0, x_min - 20)
            x_max = min(w, x_max + 20)

            try:
                # 3. Automatic Preprocessing (No SPACE bar needed)
                analysisframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                analysisframe = analysisframe[y_min:y_max, x_min:x_max]
                analysisframe = cv2.resize(analysisframe, (28, 28))

                # Prepare pixel data
                pixeldata = analysisframe.reshape(1, 28, 28, 1)
                pixeldata = pixeldata / 255.0

                # Prediction
                prediction = model.predict(pixeldata, verbose=0)
                pred_index = np.argmax(prediction[0])
                confidence = prediction[0][pred_index]

                if confidence > 0.8: # Only accept if confidence is high
                    current_letter = letterpred[pred_index]
                    
                    # Logic: Only add letter if it stays the same for 15 frames 
                    # and is different from the last added letter
                    if current_letter == last_predicted_letter:
                        frames_consistent += 1
                    else:
                        frames_consistent = 0
                        last_predicted_letter = current_letter

                    if frames_consistent == 15:
                        translated_text += current_letter
                        print(f"Added: {current_letter} | Current Translation: {translated_text}")

                # Optional: Draw on frame for visual feedback
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, last_predicted_letter, (x_min, y_min-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                pass

    # Show the video being processed
    cv2.imshow("Translating Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n--- FINAL TRANSLATION ---")
print(translated_text)