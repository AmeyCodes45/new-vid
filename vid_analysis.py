import os
import gdown
from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from collections import Counter

app = Flask(__name__)

# Ensure model exists before loading
FER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_model.h5")
if not os.path.exists(FER_MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {FER_MODEL_PATH}")

emotion_model = load_model(FER_MODEL_PATH)
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Video Analysis API is running!"})

@app.route("/analyze-video", methods=["POST"])
def analyze_video_route():
    data = request.json
    video_link = data.get("video_link")

    if not video_link:
        return jsonify({"error": "Missing video link"}), 400

    try:
        video_path = download_video(video_link)
        video_analysis = analyze_video(video_path)
        os.remove(video_path)  # Clean up temp file
        return jsonify(video_analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def download_video(url):
    output_path = "temp_video.mp4"
    try:
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        if not os.path.exists(output_path):
            raise ValueError("Failed to download video.")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error downloading video: {e}")

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h_frame, w_frame, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * w_frame), int(bboxC.ymin * h_frame), int(bboxC.width * w_frame), int(bboxC.height * h_frame)
                x, y, w, h = max(0, x), max(0, y), min(w, w_frame - x), min(h, h_frame - y)
                
                face_crop = frame[y:y + h, x:x + w]
                if face_crop.size == 0:
                    continue  # Skip if crop is empty
                
                face_crop = cv2.resize(face_crop, (48, 48))
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_crop = face_crop / 255.0
                face_crop = np.reshape(face_crop, (1, 48, 48, 1))

                prediction = emotion_model.predict(face_crop)
                emotion_label = EMOTION_LABELS[np.argmax(prediction)]
                emotions.append(emotion_label)

    cap.release()
    return {"most_frequent_emotion": Counter(emotions).most_common(1)[0][0] if emotions else "Neutral"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
