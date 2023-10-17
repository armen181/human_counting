import torch

from func.detectionfunc import detectionFunc
from func.trackfunc import trackFunc
from tracker import DeepSort


def init_model(model):
    detector = torch.hub.load('ultralytics/yolov5', "custom", path=model)
    detector.eval()
    return detector


class humanDetector():
    def __init__(self, threshold):
        self.detectr = init_model("./model/yolov5s.onnx")
        self.threshold = threshold

    def get(self, frame):
        return detectionFunc(self.detectr, frame, self.threshold)

    def release(self):
        return True


class tracking():
    def __init__(self):
        self.deepsort = DeepSort("./model/original_ckpt.onnx")

    def get(self, frame, boxes):
        return trackFunc(self.deepsort, frame, boxes)

    def release(self):
        return True
