import joblib
import cv2
import mediapipe as mp
import pandas as pd
'''
1. Open webcam
2. Load MediaPipe hand landmarker
3. Load your saved KNN model
4. Read one frame from webcam
5. Detect hand landmarks
6. Turn landmarks into the same 63-number list
7. Give those 63 numbers to the model
8. Model predicts A, B, or C
9. Show prediction on the screen
'''




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
model = joblib.load("asl_knn_model.joblib")

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

    


    if hand_landmarker_result.hand_landmarks:
        first_hand = hand_landmarker_result.hand_landmarks[0]

        let_list = []

        for landmark in first_hand:
            let_list.append(landmark.x)
            let_list.append(landmark.y)
            let_list.append(landmark.z)

        prediction = model.predict([let_list]) # predict one hand sample
        predicted_letter = prediction[0] # get the actual letter
        cv2.putText(frame, predicted_letter, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),3) #write the predicted letter on the webcam frame
        print(predicted_letter)
    else:
        print("No hand detected, nothing saved")

    # Show webcam frame
    cv2.imshow('ASL Project', frame)
    
    # Check keyboard input once
    key = cv2.waitKey(1) & 0xFF

    # If I press q, quit
    if key == ord('q'):
        break

# 4. Clean up
cap.release()
cv2.destroyAllWindows()

