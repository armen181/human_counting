from box_utils import predict as face_postprocess
from centroid_tracker import CentroidTracker
from fps_limiter import FPSLimiter
from time import perf_counter
from typing import Optional
import argparse
import cv2


def main(
    use_rknn: bool,
    threshold: float,
    line: str,
    file_path: Optional[str] = None,
    api_url: Optional[str] = None,
    camera_id: Optional[str] = None,
    fps_cap: Optional[int] = None,
    web_cam: Optional[int] = None,
    hide_window: bool = True,
):
    if file_path is not None:
        cap = cv2.VideoCapture(file_path)
    elif web_cam is not None:
        cap = cv2.VideoCapture(web_cam)
    else:
        raise ValueError("Either specify file_path or web_cam, both were None")

    line = [int(num) for num in line.split(",")]

    if use_rknn:
        from rknnpool import rknnHumanDetector, rknnFaceDetector, rknnAgeDetector, rknnGenderDetector

        firstDetector = rknnHumanDetector(threshold)
        face_detector = rknnFaceDetector(threshold)
        age_detector = rknnAgeDetector(threshold)
        gender_detector = rknnGenderDetector(threshold)
    else:
        from tourchpool import humanDetector

        firstDetector = humanDetector(threshold)

    # secondTracking = CentroidTracker(line, api_url, camera_id, face_detector, age_detector, gender_detector)

    if fps_cap is not None:
        fps_limiter = FPSLimiter(fps_cap)

    frames, loopTime, initTime = 0, perf_counter(), perf_counter()
    while cap.isOpened():
        frames += 1
        ret, frame = cap.read()
        if not ret:
            break

        face_frame = cv2.resize(frame, (640, 480))
        frame = cv2.resize(frame, (640, 640))

        face_out = face_detector.get(face_frame)

        probs = face_out[0]
        boxes = face_out[1]
        probs = probs.reshape(1, -1, 2)
        boxes = boxes.reshape(1, -1, 4)
        boxes, _, probs = face_postprocess(frame.shape[1], frame.shape[0], probs, boxes, 0.5)

        for box in boxes:
            x1, y1, x2, y2 = box
            age_gender_frame = frame[y1:y2, x1:x2]
            age_gender_frame = cv2.resize(age_gender_frame, (224, 224))
            gender = gender_detector.get(age_gender_frame)
            age = age_detector.get(age_gender_frame)
            cv2.rectangle(frame, (x2, y2), (x1, y1), (255, 255, 0), 2)
            cv2.putText(
                frame,
                gender,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                age,
                (x1, y1+20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # frame, boxes = firstDetector.get(frame)

        # This is for Ardalan's debugging, dont delete :D
        # boxes = [
        #     [10 + frames * 0.55, 10 + frames, 30, 30],
        #     [100 + frames, 500 - frames * 0.4, 30, 30],
        #     [50 + frames, 510 - frames * 1.7, 30, 30],
        #     [10 + frames, 500 - frames, 30, 30],
        #     [600 - frames * 0.5, 10 + frames * 1.5, 30, 30],
        # ]
        # boxes = [box for box in boxes if 0 < box[0] < 640]

        if boxes is None or len(boxes) == 0:
            continue

        # frame, boxes = secondTracking.get(frame, boxes)

        if not hide_window:
            cv2.imshow("Human Counting", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
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

    parser.add_argument("-r", "--use_rknn", action="store_true", help="Enable RKNN usage")
    parser.add_argument("-t", "--threshold", type=float, default=0.25, help="Detection threshold value")
    parser.add_argument(
        "-l", "--line", type=str, default="10,300,630,350", help='Line coordinates in the format "start_x,start_y,end_x,end_y" as integers'
    )
    parser.add_argument("-f", "--file_path", type=str, default=None, help="Path to video file, setting this will ignore web_cam argument")
    parser.add_argument("-fps", "--fps_cap", type=int, default=None, help="FPS cap (optional)")
    parser.add_argument("-url", "--api_url", type=str, default=None, help="Servers url to post the information")
    parser.add_argument(
        "-ci", "--camera_id", type=str, default=None, help="Camera id which is registered in the server, this is used for calling the api"
    )
    parser.add_argument("-wc", "--web_cam", type=int, default=None, help="Webcam number, 0 for first webcome (optional)")
    parser.add_argument("-hw", "--hide_window", action="store_true", help="Show the video/cam (affects performance)")

    args = parser.parse_args()

    main(
        args.use_rknn, args.threshold, args.line, args.file_path, args.api_url, args.camera_id, args.fps_cap, args.web_cam, args.hide_window
    )
