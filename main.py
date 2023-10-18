import time

import cv2

cap = cv2.VideoCapture('./resource/4.m4v')
# cap = cv2.VideoCapture(1)

use_rknn = False
threshold = 0.25

if use_rknn:
    from rknnpool import rknnHumanDetector, rknnTracking

    firstDetector = rknnHumanDetector(threshold)
    secondTracking = rknnTracking()
else:
    from tourchpool import humanDetector, tracking

    firstDetector = humanDetector(threshold)
    secondTracking = tracking()

frames, loopTime, initTime = 0, time.time(), time.time()
while cap.isOpened():
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 640))

    frame, boxes = firstDetector.get(frame)

    if boxes is None or len(boxes) == 0:
        continue

    frame, boxes = secondTracking.get(frame, boxes)

    cv2.imshow('Human Counting', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frames % 30 == 0:
        print("30 average fps:\t", 30 / (time.time() - loopTime), "帧")
        loopTime = time.time()

cap.release()
cv2.destroyAllWindows()
firstDetector.release()
