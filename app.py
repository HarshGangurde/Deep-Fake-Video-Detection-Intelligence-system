from cProfile import label

from flask import Flask, request, render_template, redirect, url_for, Response
import os
import numpy as np
import tensorflow as tf
import cv2
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.vgg16 import preprocess_input
from mtcnn import MTCNN

detector = MTCNN()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = tf.keras.models.load_model(
    r"C:\Users\LENOVO\Desktop\PROJECT\DeepFake\scripts\02deepfake_detector_v2.keras"
)

#Preprocess a single frame
def preprocess_frame(frame):

    faces = detector.detect_faces(frame)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]['box']

    x = max(0, x)
    y = max(0, y)

    face = frame[y:y+h, x:x+w]

    if face.size == 0:
        return None

    face = cv2.resize(face, (128,128))
    face = face.astype(np.float32)
    face = preprocess_input(face)

    face = np.expand_dims(face, axis=0)

    return face

# Predict on video file
def predict_video_deepfake(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    frame_count = 0
    predictions = []
    max_frames = 30

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:

            processed_frame = preprocess_frame(frame)

            if processed_frame is not None:
                pred_proba = model.predict(processed_frame, verbose=0)[0][0]
                predictions.append(pred_proba)

        frame_count += 1

        if len(predictions) >= max_frames:
            break

    cap.release()

    if predictions:
        avg_proba = float(np.mean(predictions))
        if avg_proba >= 0.5:
            label = "Fake"
            confidence = avg_proba
        else:
            label = "Real"
            confidence = 1 - avg_proba

        return {"label": label, "confidence": confidence}
    else:
        return {"error": "No frames processed."}

#  Predict from webcam
def predict_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"error": "Could not access webcam."}

    predictions = []
    frame_count = 0
    max_frames = 30

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:

            processed_frame = preprocess_frame(frame)

            if processed_frame is not None:
                pred_proba = model.predict(processed_frame, verbose=0)[0][0]
                predictions.append(pred_proba)

        frame_count += 1

        if len(predictions) >= max_frames:
            break

    cap.release()

    if predictions:
        avg_proba = float(np.mean(predictions))
        if avg_proba >= 0.5:
            label = "Fake"
            confidence = avg_proba
        else:
            label = "Real"
            confidence = 1 - avg_proba

        return {"label": label, "confidence": confidence}
    else:
        return {"error": "No frames processed."}

#  Video stream generator
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


#  Routes
@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    mode = request.form.get('mode')
    
    if mode == 'upload':
        file = request.files['video']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = predict_video_deepfake(filepath, frame_skip=15)
            return render_template('result.html', result=result)
        else:
            return render_template('result.html', result={"error": "No file selected."})

    elif mode == 'webcam':
        result = predict_webcam()
        return render_template('result.html', result=result)
    
    return render_template('result.html', result={"error": "Invalid mode selected."})

if __name__ == '__main__':
    app.run(debug=True)

    ##python app.py