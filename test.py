import cv2
import numpy as np
from tensorflow.keras.models import load_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Load your trained model
model = load_model('smnist.h5')
alphabet = "ABCDEFGHIKLMNOPQRSTUVWXY" # Note: No J or Z

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    
    # Check if frame was actually grabbed
    if not ret or frame is None:
        print("Failed to grab frame. Skipping...")
        continue 

    # Now it is safe to flip and crop
    frame = cv2.flip(frame, 1)
    roi = frame[100:300, 100:300]
    
    # 3. Preprocess the ROI to match MNIST format
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)

    # 4. Predict
    prediction = model.predict(reshaped)
    label = alphabet[np.argmax(prediction)]
    
    # 5. Display
    cv2.flip(frame, 1, frame)
    cv2.putText(frame, label, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    cv2.imshow("Sign Language App", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break