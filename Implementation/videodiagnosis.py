import cv2
import mediapipe as mp
import csv
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

sc = StandardScaler()


def calculate_acceleration(v1, v2, t1, t2):
    return (v2 - v1) / (t2 - t1)


def classify_array(arr):
    total_elements = arr.size
    num_ones = np.sum(arr)
    percentage_ones = (num_ones / total_elements) * 100
    print(percentage_ones)
    if percentage_ones <= 19:
        return 1
    else:
        return 0


def diagnose_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(fps * 15)

    csv_file = open('static/csv/landmark_coordinates.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Time', 'Left_X', 'Left_Y'])

    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        frame_number = 0
        interval = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret or frame_number >= total_frames:
                break

            frame_number += 1
            interval += 0.033356683

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                toex = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * image.shape[0]
                toey = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * image.shape[1]
                csv_writer.writerow([interval, toex, toey])
            except:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

        csv_file.close()

    cap.release()

    csv_filename = 'static/csv/landmark_coordinates.csv'
    output_filename = 'static/csv/acceleration_readings.csv'

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

    accelerations_df = pd.DataFrame(accelerations, columns=['Time', 'Acceleration_x', 'Acceleration_y'])
    accelerations_df.to_csv(output_filename, index=False)

    dataset_pred = pd.read_csv('static/csv/acceleration_readings.csv')
    X_new = dataset_pred.iloc[2:, :].values
    X_new = sc.fit_transform(X_new)

    pickled = pickle.load(open('static/model/videomodel.pkl', 'rb'))
    y_pred = pickled.predict(X_new)

    input_array = y_pred
    output = classify_array(input_array)

    if output == 1:
        return "You are diagnosed with Parkinson's"
    else:
        return "You do not have Parkinson's"
