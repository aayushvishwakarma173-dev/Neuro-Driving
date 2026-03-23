"""
Drowsiness Detection — Flask Web Application
Streams processed webcam frames over MJPEG and exposes a JSON drowsiness status endpoint.
"""

import os
import threading

import cv2
import dlib
import imutils
from flask import Flask, Response, jsonify, render_template, send_from_directory
from imutils import face_utils
from scipy.spatial import distance

# ──────────────────── Flask app ────────────────────
app = Flask(__name__)

# ──────────────────── EAR calculation ────────────────────
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ──────────────────── Parameters ────────────────────
EAR_THRESHOLD = 0.25
CONSEC_FRAMES  = 10

# Shared state (thread-safe via GIL for simple bool/int)
_drowsy_flag   = 0
_is_drowsy     = False
_lock          = threading.Lock()

# ──────────────────── dlib models ────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")
)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# ──────────────────── Frame generator ────────────────────
def generate_frames():
    global _drowsy_flag, _is_drowsy

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=640)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)

            if len(faces) == 0:
                with _lock:
                    _drowsy_flag = 0
                    _is_drowsy   = False

            for face in faces:
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                leftEye  = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]

                leftEAR  = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear      = (leftEAR + rightEAR) / 2.0

                # Draw eye contours
                cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

                # Drowsiness check
                with _lock:
                    if ear < EAR_THRESHOLD:
                        _drowsy_flag += 1
                        if _drowsy_flag >= CONSEC_FRAMES:
                            _is_drowsy = True
                            cv2.putText(frame, "!! DROWSINESS ALERT !!",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (0, 0, 255), 2)
                    else:
                        _drowsy_flag = 0
                        _is_drowsy   = False

                # EAR overlay
                cv2.putText(frame, f"EAR: {ear:.2f}",
                            (480, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        cap.release()

# ──────────────────── Routes ────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with _lock:
        return jsonify({"drowsy": _is_drowsy})


@app.route("/static/alarm.mpeg")
def serve_alarm():
    return send_from_directory(BASE_DIR, "alarm.mpeg")


# ──────────────────── Entry-point ────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
