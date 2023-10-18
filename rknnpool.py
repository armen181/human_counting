from rknnlite.api import RKNNLite

from func.rknndetectionfunc import rknnDetectionFunc
from func.rknntrackfunc import rknnTrackFunc
from tracker import DeepSort


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
    def __init__(self, threshold):
        self.threshold = threshold
        self.rknn = rknnInit("./rknnModel/human.rknn", 0)

    def get(self, frame):
        return rknnDetectionFunc(self.rknn, frame, self.threshold), True

    def release(self):
        self.rknn.release()


class rknnTracking():
    def __init__(self):
        self.rknn = rknnInit("./rknnModel/ckpt.rknn", 1)
        self.deep_sort = DeepSort(self.rknn, True)

    def get(self, frame, boxes):
        return rknnTrackFunc((), frame, boxes)

    def release(self):
        return True
