# #Importing Libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import csv
import pandas as pd
import pickle
import numpy as np


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()



# #Loading ML Dataset
# dataset = pd.read_csv('final.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, -1].values
#
#
# #Preparing ML Model
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# import xgboost as xgb
# classifier = xgb.XGBClassifier(booster='dart', eta=0.6)
# classifier.fit(X_train, y_train)
#
#
# pickle.dump(classifier, open('model.pkl', 'wb'))

################################################################################################################################


#Preparing MediaPipe Setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("C:/Users/aryan/Dropbox/Study Material/Assignments/Capstone bs/MediePipe/Dataset/Cropped_Videos/blackout_output_PDFE01_1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(fps * 30)

csv_file = open('landmark_coordinates.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Time', 'Left_X', 'Left_Y'])


# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame_number >= total_frames:
            break

        frame_number += 1

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            landmarks = results.pose_landmarks.landmark
            toex = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * image.shape[0]
            toey = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * image.shape[1]
            toe = [toex,toey]
            csv_writer.writerow([frame_number / fps, toex, toey])

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

csv_file.close()


################################################################################################################################


def calculate_acceleration(v1, v2, t1, t2):
    return (v2 - v1) / (t2 - t1)


def calculate_acceleration_data():
    csv_filename = 'landmark_coordinates.csv'

    try:
        df = pd.read_csv(csv_filename, names=['Time', 'Left_X', 'Left_Y'])
    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.")
        return

    if len(df) < 2:
        print("Error: Insufficient data in CSV file.")
        return

    df['Left_X'] = pd.to_numeric(df['Left_X'], errors='coerce')
    df['Left_Y'] = pd.to_numeric(df['Left_Y'], errors='coerce')
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

    accelerations = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]

        time_interval = row['Time'] - prev_row['Time']

        vx = (row['Left_X'] - prev_row['Left_X']) / time_interval
        vy = (row['Left_Y'] - prev_row['Left_Y']) / time_interval

        ax = calculate_acceleration(prev_row['Left_X'], row['Left_X'], prev_row['Time'], row['Time'])
        ay = calculate_acceleration(prev_row['Left_Y'], row['Left_Y'], prev_row['Time'], row['Time'])

        accelerations.append((prev_row['Time'], ax, ay))

    output_filename = 'acceleration_readings.csv'
    accelerations_df = pd.DataFrame(accelerations, columns=['Time', 'Acceleration_x', 'Acceleration_y'])
    accelerations_df.to_csv(output_filename, index=False)


calculate_acceleration_data()


################################################################################################################################

dataset_pred = pd.read_csv('acceleration_readings.csv')
X_new = dataset_pred.iloc[2:, :].values
X_new = sc.fit_transform(X_new)


pickled = pickle.load(open('model.pkl', 'rb'))
y_pred = pickled.predict(X_new)


def classify_array(arr):
    total_elements = arr.size
    num_ones = np.sum(arr)
    percentage_ones = (num_ones / total_elements) * 100
    print(percentage_ones)
    if percentage_ones >= 22:
        return 1
    else:
        return 0


input_array = y_pred
output = classify_array(input_array)

if output==1:
    print("You are diagnosed with Parkinson's")
else:
    print("You do not have Parkinson's")

################################################################################################################################