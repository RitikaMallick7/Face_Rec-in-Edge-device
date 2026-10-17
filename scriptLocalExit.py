# import face_recognition
# import cv2
# import pickle
# import numpy as np
# import time
# import os
# import csv

# # --- CONFIGURATION ---
# LOG_FILE = r"log.csv"
# CAPTURE_DIR = r"captures"
# UPLOAD_COOLDOWN = 10  # Seconds to wait before logging the same person again
# Frame_rate_cooldown=88
# Frame_rate=0

# # Create the captures directory if it doesn't exist
# if not os.path.exists(CAPTURE_DIR):
#     os.makedirs(CAPTURE_DIR)

# # Initialize the CSV file with headers if it's new
# if not os.path.exists(LOG_FILE):
#     with open(LOG_FILE, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["Name", "Date", "Time", "Status", "Image_Path"])

# # Dictionary to track {name: last_log_time}
# last_seen_dict = {}

# def log_locally(frame, name):
#     """Saves image to folder and data to CSV."""
#     timestamp_str = time.strftime("%H-%M-%S")
#     date_str = time.strftime("%Y-%m-%d")
    
#     # 1. Save the Image
#     image_name = f"{name}_{date_str}_{timestamp_str}.jpg"
#     image_path = os.path.join(CAPTURE_DIR, image_name)
#     cv2.imwrite(image_path, frame)

#     # 2. Append to CSV Data File
#     with open(LOG_FILE, mode='a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([name, date_str, timestamp_str, "Entry", image_path])
    
#     print(f" [LOCAL LOG] Recorded {name} at {timestamp_str}")

# # --- DATABASE LOADING ---
# print("Loading face database...")
# with open("encoding.pickle", "rb") as f:
#     data = pickle.load(f)
#     known_face_encodings = data["encodings"]
#     known_face_names = data["names"]

# # Initialize Video
# video_capture = cv2.VideoCapture('home test 1.mp4')

# while True:
#     Frame_rate+=1
#     ret, frame = video_capture.read()
#     if not ret: break

#     raw_frame = frame.copy()
#     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
#     face_locations = face_recognition.face_locations(rgb_small_frame)
#     face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#     current_time = time.time()

#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
#         name = "Unknown"
        
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]

#         # --- LOCAL LOGGING LOGIC ---
#         # if (current_time - last_seen_dict.get(name, 0)) > UPLOAD_COOLDOWN and name!='Unknown':
#         #     log_locally(raw_frame, name)
#         #     last_seen_dict[name] = current_time

#         # UI Drawing
#         top, right, bottom, left = top*4, right*4, bottom*4, left*4
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         if name!='Unknown':
#             cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
#         else:
#             cv2.putText(frame, ' ', (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

#         # if (current_time - last_seen_dict.get(name, 0)) > UPLOAD_COOLDOWN and name!='Unknown' :
#         if (Frame_rate - last_seen_dict.get(name, 0)) > Frame_rate_cooldown and name!='Unknown' :
#             log_locally(frame, name)
#             last_seen_dict[name] = Frame_rate
#     display_frame = cv2.resize(frame, (760, 480)) # Scale down for display only
#     cv2.imshow('Face Recognition System (Local Logging)', display_frame)
#     # cv2.imshow('Face Recognition System (Local Logging)', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video_capture.release()
# cv2.destroyAllWindows()


import face_recognition
import cv2
import pickle
import numpy as np
import time
import os
import csv

# --- CONFIGURATION ---
LOG_FILE = r"log.csv"
CAPTURE_DIR = r"captures"

FRAME_SKIP = 3              # Process every 3rd frame (huge speed boost)
RESIZE_SCALE = 0.20        # Reduce frame size (0.20 = faster)
TOLERANCE = 0.55           # Matching strictness
COOLDOWN_FRAMES = 70       # Prevent duplicate logging

frame_count = 0

# Create directories
if not os.path.exists(CAPTURE_DIR):
    os.makedirs(CAPTURE_DIR)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time", "Status", "Image_Path"])

# Track last seen frame
last_seen_dict = {}

def log_locally(frame, name):
    timestamp_str = time.strftime("%H-%M-%S")
    date_str = time.strftime("%Y-%m-%d")

    image_name = f"{name}_{date_str}_{timestamp_str}.jpg"
    image_path = os.path.join(CAPTURE_DIR, image_name)
    cv2.imwrite(image_path, frame)

    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, timestamp_str, "Exit", image_path])

    print(f"[LOGGED] {name} at {timestamp_str}")

# --- LOAD DATABASE ---
print("[*] Loading face database...")
with open("encoding.pickle", "rb") as f:
    data = pickle.load(f)
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]

print(f"[*] Loaded {len(known_face_names)} faces")

# --- VIDEO ---
video_capture = cv2.VideoCapture('Vids\exit final.mp4')

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[*] End of video")
        break

    frame_count += 1


    if frame_count % FRAME_SKIP != 0:
        continue

    # Resize + RGB
    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations,
        num_jitters=1
    )

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        name = "Unknown"
        if face_distances[best_match_index] < TOLERANCE:
            name = known_face_names[best_match_index]

        # Scale back
        scale = int(1 / RESIZE_SCALE)
        top, right, bottom, left = top*scale, right*scale, bottom*scale, left*scale

        # Draw UI
        if name != "Unknown":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if (frame_count - last_seen_dict.get(name, 0)) > COOLDOWN_FRAMES:
                log_locally(frame, name)
                last_seen_dict[name] = frame_count
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

    # Display
    display_frame = cv2.resize(frame, (960, 540))
    cv2.imshow('FAST Face Recognition (CPU Optimized)', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
