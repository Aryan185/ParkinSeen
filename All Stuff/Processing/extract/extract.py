import cv2
import mediapipe as mp
import csv
import warnings
warnings.filterwarnings("ignore")


#Preparing MediaPipe Setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("C:/Users/aryan/Dropbox/Study Material/Assignments/Capstone bs/MediePipe/Dataset/Cropped_Videos/blackout_output_PDFE35_3.mp4")
#cap = cv2.VideoCapture("C:/Users/aryan/Dropbox/Study Material/Assignments/Capstone bs/MediePipe/Dataset/Videos/PDFE31_1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)


csv_file = open('blackout_output_PDFE35_3.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Time', 'Left_X', 'Left_Y'])


## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
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
            #print(toex,toey)
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



# import cv2
# import mediapipe as mp
# import csv
# import warnings
# import numpy as np
# import torch
# import albumentations as albu
# from iglovikov_helper_functions.utils.image_utils import pad, unpad
# from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
# from cloths_segmentation.pre_trained_models import create_model
#
# warnings.filterwarnings("ignore")
#
# # Preparing MediaPipe Setup
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
#
# # Check if GPU is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
#
# # Load the pre-trained segmentation model
# model = create_model("Unet_2020-10-30").to(device)
# model.eval()
#
# # Define transformation for image preprocessing
# transform = albu.Compose([albu.Normalize(p=1)], p=1)
#
# # Open video file
# video_path = "C:/Users/aryan/Dropbox/Study Material/Assignments/Capstone bs/MediePipe/Dataset/Cropped_Videos/blackout_output_PDFE19_1.mp4"
# cap = cv2.VideoCapture(video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# # Open CSV file for writing pose data
# csv_file = open('blackout_output_PDFE31_1.csv', mode='w', newline='')
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(['Time', 'Left_X', 'Left_Y'])
#
# # Setup MediaPipe instance for pose estimation
# with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
#     frame_number = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#
#         if not ret:
#             break
#
#         frame_number += 1
#
#         # Perform cloth segmentation on GPU
#         padded_frame, pads = pad(frame, factor=32, border=cv2.BORDER_CONSTANT)
#         transformed_frame = transform(image=padded_frame)["image"]
#         x = torch.unsqueeze(tensor_from_rgb_image(transformed_frame), 0).to(device)
#
#         with torch.no_grad():
#             prediction = model(x)[0][0].cpu()
#
#         mask = (prediction > 0).numpy().astype(np.uint8)
#         mask = unpad(mask, pads)
#
#         # Apply segmentation mask to the original frame
#         masked_frame = cv2.addWeighted(frame, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0)
#
#         # Perform pose estimation
#         image = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#         results = pose.process(image)
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         try:
#             # Get left foot landmark coordinates
#             landmarks = results.pose_landmarks.landmark
#             toex = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * image.shape[1]
#             toey = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * image.shape[0]
#
#             # Write pose data to CSV
#             csv_writer.writerow([frame_number / fps, toex, toey])
#
#             # Overlay pose estimation on the frame
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
#
#             cv2.imshow('Combined Feed', image)
#
#         except Exception as e:
#             print("Error:", e)
#
#         # Check for 'q' key press to exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# csv_file.close()




