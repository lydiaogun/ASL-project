import cv2
import mediapipe as mp

# 1. Set up the hand landmarker (load the model)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

landmarker = HandLandmarker.create_from_options(options)
# 2. Open the webcam
cap = cv2.VideoCapture(0)

# 3. Loop: read frame, detect hands, print landmarks, show frame
while True:
    ret, frame = cap.read()
    
    # convert frame to mediapipe image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    hand_landmarker_result = landmarker.detect(mp_image)
    # run detection
    # print the result

    # print the result
    if hand_landmarker_result.hand_landmarks:
        hand = hand_landmarker_result.hand_landmarks[0]

        index_joint = hand[6]
        index_tip = hand[8]

        middle_joint = hand[10]
        middle_tip = hand[12]

        ring_joint = hand[14]
        ring_tip = hand[16]

        pinky_joint = hand[18]
        pinky_tip = hand[20]


       

    

        if index_joint.y > index_tip.y and  middle_joint.y > middle_tip.y and ring_joint.y > ring_tip.y and pinky_joint.y > pinky_tip.y:
            print("hand is open")
        else:
            print("hand is closed")

      
    
    

    cv2.imshow('ASL Project', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Clean up
cap.release()
cv2.destroyAllWindows()