import cv2
import time

cap = cv2.VideoCapture('./resource/1.mp4')
#cap = cv2.VideoCapture(1)

use_rknn = False
threshold = 0.3

if use_rknn:
    from rknnpool import rknnHumanDetector
    firstDetector = rknnHumanDetector(threshold)
else:
    from tourchpool import humanDetector
    firstDetector = humanDetector(threshold)

frames, loopTime, initTime = 0, time.time(), time.time()
while cap.isOpened():
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 640))
    frame, flag = firstDetector.get(frame)
    if not flag:
        break
    cv2.imshow('Human Counting', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frames % 30 == 0:
        print("30 average fps:\t", 30 / (time.time() - loopTime), "帧")
        loopTime = time.time()

cap.release()
cv2.destroyAllWindows()
firstDetector.release()
