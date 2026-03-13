import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import time
import np
model_path = 'assets/pose_landmarker_full.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_frame = None
count = 1
inDownPos = False

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),       # face left
    (0, 4), (4, 5), (5, 6), (6, 8),       # face right
    (11, 12),                               # shoulders
    (11, 13), (13, 15),                    # left arm
    (12, 14), (14, 16),                    # right arm
    (11, 23), (12, 24), (23, 24),          # torso
    (23, 25), (25, 27),                    # left leg
    (24, 26), (26, 28),                    # right leg
    (27, 29), (29, 31), (27, 31),          # left foot
    (28, 30), (30, 32), (28, 32),          # right foot
]






def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_frame, count, inDownPos

    frame = cv2.cvtColor(output_image.numpy_view().copy(), cv2.COLOR_RGB2BGR)

    if result.pose_landmarks:
        h, w, _ = frame.shape
        

        

        for pose in result.pose_landmarks:
            # lines
            for (a, b) in POSE_CONNECTIONS:
                x1, y1 = int(pose[a].x * w), int(pose[a].y * h)
                x2, y2 = int(pose[b].x * w), int(pose[b].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), ( (0, 255, 0) if inDownPos else (111, 64, 125)), 2)

            # dots
            for lm in pose:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, ( (0, 255, 0) if inDownPos else (111, 64, 125)), -1)

            #pushup detection
            
            if(pose[11].y > pose[13].y and pose[12].y > pose[14].y and not inDownPos):
                count += 1
                inDownPos = True
            else:
                inDownPos = False
            

                
                
                
    latest_frame = frame  # just store it


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    min_pose_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    startTime = time.time_ns()
    while cap.isOpened():
        frame_timestamp_ms = (time.time_ns() - startTime) // 10**6

        success, image = cap.read()
        if not success:
            print("empty camera")
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        if latest_frame is not None:
            cv2.imshow("PoseLandmarker", latest_frame)  # only called from main thread 
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()