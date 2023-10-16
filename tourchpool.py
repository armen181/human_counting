import torch
from func.detectionfunc import detectionFunc

def init_model(model):
    detector = torch.hub.load('ultralytics/yolov5', "custom", path=model)
    detector.eval()
    return detector


class humanDetector():
    def __init__(self, threshold):
        self.detectr = init_model("./model/yolov5s.onnx")
        self.threshold = threshold

    def get(self, frame):
        return detectionFunc(self.detectr, frame, self.threshold), True

    def release(self):
        return True
