import cv2

def detectionFunc(detector, IMG, threshold):
    fut = detector(IMG).pandas().xyxy[0]
    for detection in fut.values:
        xmin, ymin, xmax, ymax, confidence, id = detection[:6]
        if id == 0 and confidence >= threshold:
            cv2.rectangle(IMG, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

    return IMG
