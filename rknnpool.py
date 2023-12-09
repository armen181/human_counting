from rknnlite.api import RKNNLite
import numpy as np

from func.rknndetectionfunc import rknnDetectionFunc
# from func.rknntrackfunc import rknnTrackFunc
# from tracker import DeepSort
import cv2


def rknnInit(rknnModel, id):
    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print(rknnModel, "\t\tdone")
    return rknn_lite


class rknnHumanDetector():
    def __init__(self, threshold, npu=0):
        self.threshold = threshold
        self.rknn = rknnInit("./rknnModel/human.rknn", npu)

    def get(self, frame):
        return rknnDetectionFunc(self.rknn, frame, self.threshold)

    def release(self):
        self.rknn.release()


class rknnFaceDetector():
    def __init__(self, threshold, npu=1):
        self.threshold = threshold
        self.rknn = rknnInit("./rknnModel/face.rknn", npu)

    def get(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.reshape(frame, (3, frame.shape[0], frame.shape[1]))
        outputs = self.rknn.inference(inputs=[frame])
        return outputs

    def release(self):
        self.rknn.release()


class rknnAgeDetector():
    def __init__(self, threshold, npu=2):
        self.threshold = threshold
        self.rknn = rknnInit("./rknnModel/age.rknn", npu)

    def get(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = self.rknn.inference(inputs=[frame])
        return outputs


    def release(self):
        self.rknn.release()

class rknnGenderDetector():
    def __init__(self, threshold, npu=2):
        self.threshold = threshold
        self.rknn = rknnInit("./rknnModel/gender.rknn", npu)

    def get(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = self.rknn.inference(inputs=[frame])
        return outputs


    def release(self):
        self.rknn.release()


# class rknnTracking():
#     def __init__(self):
#         self.rknn = rknnInit("./rknnModel/ckpt.rknn", 1)
#         self.deep_sort = DeepSort(self.rknn, True)

#     def get(self, frame, boxes):
#         return rknnTrackFunc(self.deep_sort, frame, boxes)

#     def release(self):
#         return True
