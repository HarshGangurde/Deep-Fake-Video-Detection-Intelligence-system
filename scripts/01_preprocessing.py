import os
import cv2
import numpy as np
from mtcnn import MTCNN

# Initialize face detector once (faster)
detector = MTCNN()

# Preprocessing function
def preprocess_video(input_dir, output_dir, filename, img_size=(224, 224), frame_skip=20):

    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(input_dir, filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Couldn't open video {filename}")
        return

    frame_count = 0
    frame_index = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # Process only every Nth frame
        if frame_index % frame_skip == 0:

            faces = detector.detect_faces(frame)

            for face in faces:

                x, y, w, h = face['box']

                # Ensure coordinates are positive
                x = max(0, x)
                y = max(0, y)

                # Crop the face
                face_crop = frame[y:y+h, x:x+w]

                if face_crop.size == 0:
                    continue

                # Resize face
                face_crop = cv2.resize(face_crop, img_size)

                # Normalize
                face_crop = face_crop / 255.0

                # Save image
                output_filename = f"{filename}_frame_{frame_count}.jpg"

                cv2.imwrite(
                    os.path.join(output_dir, output_filename),
                    (face_crop * 255).astype(np.uint8)
                )

                frame_count += 1

        frame_index += 1

    cap.release()


def process_videos(input_dir, output_dir, frame_skip=20):
    os.makedirs(output_dir, exist_ok=True)   # create folder if missing
    filenames = os.listdir(input_dir)[:200]

    for filename in filenames:

        if filename.endswith((".mp4", ".avi", ".mov")):

            # Skip if already processed
            existing = [f for f in os.listdir(output_dir) if filename in f]
            if len(existing) > 0:
                print(f"Skipping already processed: {filename}")
                continue

            print(f"Processing video: {filename}")

            preprocess_video(input_dir, output_dir, filename, frame_skip=frame_skip)


if __name__ == "__main__":

    process_videos('data/raw_data/deepfake', 'data/processed/deepfake', frame_skip=20)
    process_videos('data/raw_data/real', 'data/processed/real', frame_skip=20)