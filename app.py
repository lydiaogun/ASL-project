from flask import Flask, request, jsonify
from landmarks import normalise
import mediapipe as mp
import numpy as np
import cv2
import joblib

my_custom_classifier = joblib.load("asl_knn_model.joblib")


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

landmarker = HandLandmarker.create_from_options(options)
app = Flask(__name__)

@app.route("/predict", methods  = ['POST'])
def predict():
        
    if 'image' not in request.files:
        return jsonify({"error": "No image found"}), 400
    # Get the file from the request
    file = request.files['image']
    file_bytes = file.read()
    # Convert raw bytes to a 1D NumPy uint8 array
    img_buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    
    #Decode buffer into an OpenCV BGR image
    bgr_image = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
    
    #Convert BGR to RGB (Crucial for mp.ImageFormat.SRGB)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    #THE BRIDGE: Wrap the RGB NumPy array into a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    result = landmarker.detect(mp_image)
    if result.hand_landmarks:
        first_hand = result.hand_landmarks[0]
        handed = result.handedness[0][0].display_name

        let_list = []

        for landmark in first_hand:
            let_list.append(landmark.x)
            let_list.append(landmark.y)
            let_list.append(landmark.z)
        nlist = normalise(let_list, handed)
        prediction_batch = my_custom_classifier.predict([nlist])
        predicton_result = prediction_batch[0]
        return jsonify({"letter" : predicton_result})
    else:
        return jsonify({"error" :  "No letter was found"}), 400
        

            


            