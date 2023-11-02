from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
 

class CentroidTracker:
    def __init__(self, maxDisappeared: int = 50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
 
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1
 
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
 
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
 
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
 
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
 
            rows = D.min(axis=1).argsort()
 
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
 
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
 
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
 
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
 
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

    def get(self, frame, boxes):
        rects = [[x - w//2, y - h//2, x + w//2, y + h//2] for x, y, w, h in boxes]
        self.update(rects)

        for bbox in rects:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

        for _id, (x, y) in self.objects.items():
            cv2.putText(frame, str(_id), (x - 7, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

        return frame, boxes