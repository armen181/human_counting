from centroid_tracker import CentroidTracker
from time import perf_counter, sleep
from fps_limiter import FPSLimiter
from typing import Optional
import argparse
import cv2


def main(
    use_rknn: bool,
    threshold: float,
    file_path: Optional[str] = None,
    fps_cap: Optional[int] = None,
    web_cam: Optional[int] = None,
    hide_window: bool = True):

    if file_path is not None:
        cap = cv2.VideoCapture(file_path)
    elif web_cam is not None:
        cap = cv2.VideoCapture(web_cam)
    else:
        raise ValueError("Either specify file_path or web_cam, both were None")

    if use_rknn:
        from rknnpool import rknnHumanDetector
        firstDetector = rknnHumanDetector(threshold)
    else:
        from tourchpool import humanDetector
        firstDetector = humanDetector(threshold)

    secondTracking = CentroidTracker()

    if fps_cap is not None:
        fps_limiter = FPSLimiter(fps_cap)

    frames, loopTime, initTime = 0, perf_counter(), perf_counter()
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

        if not hide_window:
            cv2.imshow('Human Counting', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frames % 30 == 0:
            print("30 average fps:\t", 30 / (perf_counter() - loopTime), "帧")
            loopTime = perf_counter()

        if fps_cap is not None:
            fps_limiter.update()

    cap.release()
    cv2.destroyAllWindows()
    firstDetector.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for human detection")

    parser.add_argument('-r', '--use_rknn', action='store_true', help="Enable RKNN usage")
    parser.add_argument('-t', '--threshold', type=float, default=0.25, help="Detection threshold value")
    parser.add_argument('-f', '--file_path', type=str, default=None, help="Path to video file, setting this will ignore web_cam argument")
    parser.add_argument('-fps', '--fps_cap', type=int, default=None, help="FPS cap (optional)")
    parser.add_argument('-wc', '--web_cam', type=int, default=None, help="Webcam number, 0 for first webcome (optional)")
    parser.add_argument('-hw', '--hide_window', action='store_true', help="Show the video/cam (affects performance)")

    args = parser.parse_args()

    main(args.use_rknn, args.threshold, args.file_path, args.fps_cap, args.web_cam, args.hide_window)

