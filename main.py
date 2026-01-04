import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pandas as pd
from mediapipe.modules import landmark_pb2

model = load_model('smnist.h5')


def compute_bbox(hand_landmarks, width, height, padding=20):
    x_coords = [int(lm.x * width) for lm in hand_landmarks]
    y_coords = [int(lm.y * height) for lm in hand_landmarks]
    x_min = max(min(x_coords) - padding, 0)
    y_min = max(min(y_coords) - padding, 0)
    x_max = min(max(x_coords) + padding, width)
    y_max = min(max(y_coords) + padding, height)
    if x_min >= x_max or y_min >= y_max:
        return None
    return x_min, y_min, x_max, y_max

mp_tasks = mp.tasks
mp_vision = mp.tasks.vision
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
landmarker = mp_vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError('Unable to access camera 0')

img_counter = 0
analysisframe = ''
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
while True:
    ret, frame = cap.read()
    if not ret:
        print('Unable to read from camera. Exiting.')
        break

    h, w = frame.shape[:2]

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        analysisframe = frame
        cv2.imshow("Frame", analysisframe)
        framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=framergbanalysis)
        resultanalysis = landmarker.detect(mp_image)
        hand_landmarksanalysis = resultanalysis.hand_landmarks
        bbox = None
        if hand_landmarksanalysis:
            for handLMsanalysis in hand_landmarksanalysis:
                bbox = compute_bbox(handLMsanalysis, w, h)
                if bbox:
                    break
        if not bbox:
            print('No hand detected. Press space again to retry.')
            continue

        x_min, y_min, x_max, y_max = bbox
        analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
        analysisframe = analysisframe[y_min:y_max, x_min:x_max]
        analysisframe = cv2.resize(analysisframe,(28,28))


        nlist = []
        rows,cols = analysisframe.shape
        for i in range(rows):
            for j in range(cols):
                k = analysisframe[i,j]
                nlist.append(k)
        
        datan = pd.DataFrame(nlist).T
        colname = []
        for val in range(784):
            colname.append(val)
        datan.columns = colname

        pixeldata = datan.values
        pixeldata = pixeldata / 255
        pixeldata = pixeldata.reshape(-1,28,28,1)
        prediction = model.predict(pixeldata)
        predarray = np.array(prediction[0])
        letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
        predarrayordered = sorted(predarray, reverse=True)
        high1 = predarrayordered[0]
        high2 = predarrayordered[1]
        high3 = predarrayordered[2]
        for key,value in letter_prediction_dict.items():
            if value==high1:
                print("Predicted Character 1: ", key)
                print('Confidence 1: ', 100*value)
            elif value==high2:
                print("Predicted Character 2: ", key)
                print('Confidence 2: ', 100*value)
            elif value==high3:
                print("Predicted Character 3: ", key)
                print('Confidence 3: ', 100*value)
        time.sleep(5)

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=framergb)
    result = landmarker.detect(mp_image)
    hand_landmarks = result.hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            bbox = compute_bbox(handLMs, w, h)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=handLMs)
            mp_drawing.draw_landmarks(frame, landmark_list, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()