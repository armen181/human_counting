import cv2


def rknnTrackFunc(deepsort, frame, boxes):
    tracked_boxes = deepsort.update(boxes, frame)

    for bbox in tracked_boxes:
        x, y, w, h, trk_id = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, str(trk_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame, tracked_boxes
