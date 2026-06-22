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
        wrist = hand_landmarker_result.hand_landmarks[0][0]
        print(f"Wrist position: x={wrist.x:.3f}, y={wrist.y:.3f}, z={wrist.z:.3f}")
    
    cv2.imshow('ASL Project', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Clean up
cap.release()
cv2.destroyAllWindows()