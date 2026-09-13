import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)
with HandLandmarker.create_from_options(options) as landmarker:
    #the 24 letter list
    alpha_list = os.listdir('asl_alphabet_train/asl_alphabet_train')


    #write the headder
    if not os.path.exists('dataset_kaggle.csv') :
        header = []
        for i in range(21):
            header.append("x" + str(i))
            header.append("y" + str(i))
            header.append("z"+ str(i))
        header.append("L/R")
        header.append("label")
        with open('dataset_kaggle.csv', 'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    hit = 0 
    miss = 0
    total = 0
    for letter in alpha_list:
        path = os.path.join("asl_alphabet_train/asl_alphabet_train", letter)
        filenames = os.listdir(path)[::10]
        hit = 0
        miss = 0
        for filename in filenames:
            image_path = os.path.join(path , filename)
            mp_image = mp.Image.create_from_file(image_path)
            result = landmarker.detect(mp_image)
            if result.hand_landmarks:
                first_hand = result.hand_landmarks[0]
                handed = result.handedness[0][0].display_name

                let_list = []

                for landmark in first_hand:
                    let_list.append(landmark.x)
                    let_list.append(landmark.y)
                    let_list.append(landmark.z)
                let_list.append(handed)
                let_list.append(letter)
                with open('dataset_kaggle.csv', 'a',newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(let_list)
                hit+=1
            else:
                miss +=1
            total += 1
        print(letter, hit,miss)
    print(total)
        


    
 
    
