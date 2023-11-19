from typing import List, Tuple, Dict, Optional
from scipy.spatial import distance as dist
from datetime import datetime, timezone
import numpy as np
import cv2


class TrackedObject:
    next_instance_id: int = 0

    def __init__(self, center: Tuple[int, int], line: Tuple[Tuple[int, int], Tuple[int, int]]) -> None:
        self.id: int = TrackedObject.next_instance_id
        TrackedObject.next_instance_id += 1
        self.centers: List[Tuple[int, int]] = []
        self.disappeared = 0
        self.created_datetime = datetime.now(timezone.utc)
        self.line = line
        self.add_instance(center)

    def add_instance(self, center: Tuple[int, int]):
        self.centers.append(center)
        self.disappeared = 0

    def does_cross_the_line(self) -> Tuple[bool, Optional[bool]]:
        is_crossed = False
        direction = None

        if len(self.centers) > 1:
            is_crossed, direction = self._intersect(self.centers[-2], self.centers[-1], self.line)

        return is_crossed, direction

    def increase_disappeared(self):
        self.disappeared += 1

    def get_center(self) -> Tuple[int, int]:
        return self.centers[-1]

    def deregister(self):
        pass

    @staticmethod
    def _ccw(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int]):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @staticmethod
    def _intersect(
        A: Tuple[int, int],
        B: Tuple[int, int],
        line: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> Tuple[bool, Optional[bool]]:
        C, D = line
        TO = TrackedObject
        res = TO._ccw(A, C, D) != TO._ccw(B, C, D) and TO._ccw(A, B, C) != TO._ccw(A, B, D)
        direction = None

        if res:
            direction = TO._ccw(B, C, D)
            print(TO._ccw(A, C, D), TO._ccw(B, C, D), TO._ccw(A, B, C), TO._ccw(A, B, D))

        return res, direction


class CentroidTracker:
    def __init__(self, line: Tuple[Tuple[int, int], Tuple[int, int]], maxDisappeared: int = 50):
        self.objects: Dict[TrackedObject] = {}
        self.maxDisappeared = maxDisappeared
        self.line = line
        self.num_ins = 0
        self.num_outs = 0

    def register(self, centroid: Tuple[int, int]):
        new_obj = TrackedObject(centroid, self.line)
        self.objects[new_obj.id] = new_obj

    def deregister(self, objectID):
        del self.objects[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for tracked_objct in self.objects.values():
                tracked_objct.increase_disappeared()
                if tracked_objct.disappeared > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list([obj.get_center() for obj in self.objects.values()])

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID].add_instance(inputCentroids[col])

                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.objects[objectID].increase_disappeared()

                    if self.objects[objectID].disappeared > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

    def get(self, frame, boxes):
        cv2.line(frame, self.line[0], self.line[1], (0, 255, 0), 2)

        rects = [[x - w // 2, y - h // 2, x + w // 2, y + h // 2] for x, y, w, h in boxes]
        self.update(rects)

        for bbox in rects:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

        for _id, obj in self.objects.items():
            intersect, direction = obj.does_cross_the_line()
            if intersect:
                if direction:
                    self.num_ins += 1
                else:
                    self.num_outs += 1

            (x, y) = obj.get_center()
            cv2.putText(
                frame,
                str(_id),
                (x - 7, y + 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

        cv2.putText(
            frame,
            f"In: {self.num_ins}",

            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Out: {self.num_outs}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        return frame, boxes
