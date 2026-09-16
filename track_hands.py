import cv2
import mediapipe as mp
import csv
import pandas as pd
import string
import os

# 1. Set up the hand landmarker (load the model)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = HandLandmarker.create_from_options(options)

# 2. Open the webcam
cap = cv2.VideoCapture(0)
if not os.path.exists('dataset.csv') :

    #headder for csv
    header = []
    for i in range(21):
        header.append("x" + str(i))
        header.append("y" + str(i))
        header.append("z"+ str(i))
    header.append("L/R")
    header.append("label")
    with open('dataset.csv', 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)



# 3. Loop: read frame, detect hands, print landmarks, show frame
while True:
    ret, frame = cap.read() #Take one picture from the webcam right now.

    if not ret:
        print("Could not read from webcam")
        break

    # Convert frame to MediaPipe image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Look at this image and tell me if you see a hand
    hand_landmarker_result = landmarker.detect(mp_image) #Look at this one picture/frame and detect the hand landmarks.
 
    # Show webcam frame
    cv2.imshow('ASL Project', frame)

  

    # Check keyboard input once
    key = cv2.waitKey(1) & 0xFF
    


    
    if key == 27:
        break

    # If I press s, capture the current hand landmarks
    lowlet = (chr(key)).lower()
    if lowlet in string.ascii_lowercase:

        if hand_landmarker_result.hand_landmarks:
            first_hand = hand_landmarker_result.hand_landmarks[0]
            handed = hand_landmarker_result.handedness[0][0].display_name

            let_list = []

            for landmark in first_hand:
                let_list.append(landmark.x)
                let_list.append(landmark.y)
                let_list.append(landmark.z)
            let_list.append(handed)
            let_list.append(lowlet)
            with open('dataset.csv', 'a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(let_list)

            print(let_list)
            print(f"Saved one sample with {len(let_list)} values")

        else:
            print("No hand detected, nothing saved")

    

# 4. Clean up
cap.release()
cv2.destroyAllWindows()