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

latest_frame = None
count = 0

mode = None
initializing_start_time = None

foot_pos = None


POSE_CONNECTIONS = [
    #(0, 1), (1, 2), (2, 3), (3, 7),       # face left
    #(0, 4), (4, 5), (5, 6), (6, 8),       # face right
    (11, 12),                               # shoulders
    (11, 13), (13, 15),                    # left arm
    (12, 14), (14, 16),                    # right arm
    (11, 23), (12, 24), (23, 24),          # torso
    (23, 25), (25, 27),                    # left leg
    (24, 26), (26, 28),                    # right leg
    (27, 29), (29, 31), (27, 31),          # left foot
    (28, 30), (30, 32), (28, 32),          # right foot
]


GESTURE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3,4), #thumb
    
    (0, 5), (5, 6), (6, 7), (7, 8),        # pointer
    (9, 10), (10, 11), (11,12),            # middle
    (13, 14), (14, 15), (15, 16),          # ring
    (17, 18), (18, 19), (19, 20),          # pinky
    (3, 5), (5, 9), (9, 13), (13, 17)

]





def getAngle(a, b, c):
    
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def handlePose(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    return
def handleGesture(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_frame, foot_pos, initializing_start_time, mode


    frame = cv2.cvtColor(output_image.numpy_view().copy(), cv2.COLOR_RGB2BGR)
    if result.gestures:
        top_gesture = result.gestures[0][0].category_name

        h, w, _ = frame.shape


        initializing = mode == "initializing"
        if top_gesture == "Thumb_Up":
            if(not initializing):

                
                mode = "initializing"
                initializing_start_time = timestamp_ms
            else:
                if (timestamp_ms - initializing_start_time) / 10**9 >= 3:
                    mode = "stepping"
                    
                print((timestamp_ms - initializing_start_time) / 10**3)
        else:
            mode = None
            initializing_start_time = None

        



        for gesture in result.hand_landmarks:
            for (a, b) in GESTURE_CONNECTIONS:
                x1, y1 = int(gesture[a].x * w), int(gesture[a].y * h)
                x2, y2 = int(gesture[b].x * w), int(gesture[b].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), ( (0, 255, 0) if initializing else ( (11, 64, 125))), 2)

                # dots
                for lm in gesture:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, ( (0, 255, 0) if initializing else ( (11, 64, 125))), -1)
    
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
    result_callback=handleGesture)

cap = cv2.VideoCapture(0)

with GestureRecognizer.create_from_options(gesture_options) as recognizer:
    startTime = time.time_ns()
    
    while cap.isOpened():
        
        frame_timestamp_ms = (time.time_ns() - startTime) // 10**6

        success, image = cap.read()
        if not success:
            print("empty camera")
            break
        
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        recognizer.recognize_async(mp_image, frame_timestamp_ms)
        
        if latest_frame is not None:
            cv2.imshow("Gesture", latest_frame)  # only called from main thread 
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()