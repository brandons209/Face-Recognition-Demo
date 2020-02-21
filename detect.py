import face_recognition
import cv2
import numpy as np
import glob
import os
import sys
from threading import Thread

NUM_PICS = 0

def check_folder():
    files = list(filter(os.path.isfile, glob.glob(os.path.join("faces", "*.png"))))
    files.sort(key=lambda x: os.path.getmtime(x))
    global NUM_PICS

    if len(files) == NUM_PICS:
        return [], []
    NUM_PICS = len(files)
    names = [os.path.split(f)[1] for f in files]
    names = [s.replace(".png", "") for s in names]

    encodings = []
    for file in files:
        img = face_recognition.load_image_file(file)
        encoding = face_recognition.face_encodings(img)[0]
        encodings.append(encoding)

    return names, encodings

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cam_index>")
        sys.exit(1)

    cam_index = int(sys.argv[1])
    # get video from USB webcam
    video_capture = cv2.VideoCapture(cam_index)
    video_capture.set(3, 960)
    video_capture.set(4, 540)

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    names, known_encodings = check_folder()

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 1920, 1080)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        cur_names, cur_encodings = check_folder()
        if cur_names:
            names, known_encodings = cur_names, cur_encodings

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if face_distances.any():
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = names[best_match_index]

            face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 60% size
            top = int(top * 10/6)
            right = int(right * 10/6)
            bottom = int(bottom * 10/6)
            left = int(left * 10/6)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name ABOVE the face
            cv2.rectangle(frame, (left, top - 25), (right, top), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, top - 6), font, 0.8, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
