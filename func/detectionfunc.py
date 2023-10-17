def detectionFunc(detector, IMG, threshold):
    fut = detector(IMG).pandas().xyxy[0]
    boxes = []
    for detection in fut.values:
        xmin, ymin, xmax, ymax, confidence, id = detection[:6]
        if id == 0 and confidence >= threshold:
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            w = xmax - xmin
            h = ymax - ymin
            boxes.append([cx, cy, w, h])

    return IMG, boxes
