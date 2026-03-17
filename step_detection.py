import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import time
import numpy as np

pose_model_path = 'assets/pose_landmarker_full.task'
gesture_model_path = 'assets/gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

base_frame = None
latest_frame = None
mode = None
initializing_start_time = None
foot_pos = None
knee_pos = None
step_text = ""
left_foot_up = False
right_foot_up = False

POSE_CONNECTIONS = [
    (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32),
]

GESTURE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (3, 5), (5, 9), (9, 13), (13, 17)
]


def handlePose(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global base_frame, foot_pos, knee_pos, mode, step_text, left_foot_up, right_foot_up

    frame = cv2.cvtColor(output_image.numpy_view().copy(), cv2.COLOR_RGB2BGR)
    h, w, _ = frame.shape

    if result.pose_landmarks:
        pose = result.pose_landmarks[0]

        left_foot_y = pose[31].y
        right_foot_y = pose[32].y
        left_knee_y = pose[25].y
        right_knee_y = pose[26].y

        if mode == "initializing" and isinstance(foot_pos, float):
            alpha = 0.1
            avg_foot = (left_foot_y + right_foot_y) / 2
            avg_knee = (left_knee_y + right_knee_y) / 2

            if foot_pos == 0.0:
                foot_pos = avg_foot
                knee_pos = avg_knee
            else:
                foot_pos = alpha * avg_foot + (1 - alpha) * foot_pos
                knee_pos = alpha * avg_knee + (1 - alpha) * knee_pos

        elif mode == "stepping" and isinstance(foot_pos, float):

            threshold = 3 * (knee_pos + foot_pos) / 4
            lift_threshold = threshold - 0.02
            ground_threshold = threshold + 0.02

            if not left_foot_up and left_foot_y < lift_threshold:
                left_foot_up = True
            elif left_foot_up and left_foot_y > ground_threshold:
                step_text = "Left Step"
                print("Left Step")
                print(round(left_foot_y, 4), round(threshold, 4))
                left_foot_up = False

            if not right_foot_up and right_foot_y < lift_threshold:
                right_foot_up = True
            elif right_foot_up and right_foot_y > ground_threshold:
                step_text = "Right Step"
                print("Right Step")
                print(round(right_foot_y, 4), round(threshold, 4))
                right_foot_up = False

    base_frame = frame



def handleGesture(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_frame, base_frame, foot_pos, knee_pos, initializing_start_time, mode, step_text, left_foot_up, right_foot_up

    if base_frame is not None:
        frame = base_frame.copy()
    else:
        frame = cv2.cvtColor(output_image.numpy_view().copy(), cv2.COLOR_RGB2BGR)

    h, w, _ = frame.shape

    if result.gestures:
        top_gesture = result.gestures[0][0].category_name

        if top_gesture == "Thumb_Up":
            if mode != "initializing" and mode != "stepping":
                mode = "initializing"
                foot_pos = 0.0
                knee_pos = 0.0
                initializing_start_time = timestamp_ms

            elif mode == "initializing":
                elapsed = (timestamp_ms - initializing_start_time) / 1000
                if elapsed >= 3:
                    mode = "stepping"
                    initializing_start_time = None
                    print(f"Calibrated — foot: {foot_pos:}, knee: {knee_pos:}")

        for hand in result.hand_landmarks:
            color = (0, 255, 0) if mode == "initializing" else (111, 64, 125)
            for (a, b) in GESTURE_CONNECTIONS:
                x1, y1 = int(hand[a].x * w), int(hand[a].y * h)
                x2, y2 = int(hand[b].x * w), int(hand[b].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), color, 3)
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, color, -1)

    cv2.putText(frame, f'Mode: {mode}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if isinstance(foot_pos, float):
        cv2.putText(frame, f'Foot Y: {foot_pos:.4f}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    if step_text:
        cv2.putText(frame, step_text, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    latest_frame = frame


pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handlePose,
    min_pose_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=gesture_model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handleGesture
)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
     GestureRecognizer.create_from_options(gesture_options) as recognizer:

    startTime = time.time_ns()

    while cap.isOpened():
        frame_timestamp_ms = (time.time_ns() - startTime) // 10**6

        success, image = cap.read()
        if not success:
            print("empty camera")
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        pose_landmarker.detect_async(mp_image, frame_timestamp_ms)
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        if latest_frame is not None:
            cv2.imshow("Step Detection", latest_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()