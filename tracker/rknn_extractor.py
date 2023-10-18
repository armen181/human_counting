import cv2
import numpy as np


def preprocess(img):
    img = cv2.resize(img, (128, 64))
    img = np.float32(img)
    img = img / 255.0
    img = img.transpose(2, 1, 0)
    img = np.expand_dims(img, axis=0)

    return img


class RknnExtractor:
    def __init__(self, model) -> None:
        self.onnx_model = model
        self.input_names = ["input"]
        self.output_names = ["output"]

    def __call__(self, im_crops):
        embs = []
        for im in im_crops:
            inp = preprocess(im)
            emb = self.onnx_model.inference(inputs=[im_crops])
            # emb = self.onnx_model.run(self.output_names, {self.input_names[0]: inp})[0]
            embs.append(emb.squeeze())
        embs = np.array(np.stack(embs), dtype=np.float32)
        return embs
