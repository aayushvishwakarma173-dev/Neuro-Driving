from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from playsound import playsound
import threading

# ------------------ EAR FUNCTION ------------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ------------------ ALARM FUNCTION ------------------
alarm_on = False

def play_alarm():
    global alarm_on
    if not alarm_on:
        alarm_on = True
        playsound("alarm.mpeg")
        alarm_on = False

# ------------------ PARAMETERS ------------------
thresh = 0.25
frame_check = 20
flag = 0

# ------------------ DLIB MODELS ------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat"
)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# ------------------ CAMERA ------------------
cap = cv2.VideoCapture(0)

# ------------------ MAIN LOOP ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    if len(faces) == 0:
        flag = 0

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # ------------------ DROWSINESS CHECK ------------------
        if ear < thresh:
            flag += 1
            print("Drowsy Frames:", flag)

            if flag >= frame_check:
                threading.Thread(
                    target=play_alarm, daemon=True
                ).start()

                cv2.putText(
                    frame, "************ ALERT ************",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )

                cv2.putText(
                    frame, "************ ALERT ************",
                    (10, 325),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )
        else:
            flag = 0

        # Show EAR value (optional but good)
        cv2.putText(
            frame, f"EAR: {ear:.2f}",
            (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

    # ------------------ DISPLAY ------------------
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# ------------------ CLEANUP ------------------
cv2.destroyAllWindows()
cap.release()