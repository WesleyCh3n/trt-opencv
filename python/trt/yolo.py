import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from trt.engine import TensorRTInfer
from trt.utils import batch


class YOLOv8:
    def __init__(
        self,
        model_path: str,
        batch_size: int,
        confidence_threshold=0.25,
        nms_threshold=0.7,
    ):
        self.bs = batch_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = TensorRTInfer(model_path, batch_size)
        self.intput_spec, _ = self.model.input_spec()
        self.output_spec, _ = self.model.output_spec()

    def __call__(self, imgs: list) -> list:
        blobs = np.zeros((len(imgs), *self.intput_spec[1:]), dtype=np.float32)
        shapes = np.zeros((len(imgs), 2), dtype=np.int32)

        num_workers = os.cpu_count()
        if num_workers is None:
            raise RuntimeError("Unable to determine number of CPU cores")
        executor = ThreadPoolExecutor(int(num_workers / 2))
        jobs = [executor.submit(self.preprocess, img, i) for i, img in enumerate(imgs)]
        for job in as_completed(jobs):
            blob, shape, i = job.result()
            shapes[i] = shape
            blobs[i] = blob

        result = np.zeros((len(imgs), *self.output_spec[1:]), dtype=np.float32)
        for i, b in enumerate(batch(blobs, self.bs)):
            result[self.bs * i : self.bs * (i + 1)] = self.model.infer(b, len(b))

        return self.postprocess(result, shapes)

    def preprocess(self, image_path: str, index: int):
        img = cv2.imread(image_path)
        shape = img.shape
        img = self.letterbox(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))  # to CxWxH
        img = img.astype(np.float32) / 255.0
        blob = np.expand_dims(img, axis=0)  # Add batch dimension
        blob = np.ascontiguousarray(blob, np.float32)
        return blob, (shape[0], shape[1]), index

    def letterbox(self, img: np.ndarray) -> np.ndarray:
        scale = min(
            (self.intput_spec[2] / img.shape[0]), (self.intput_spec[3] / img.shape[1])
        )
        new_h = round(img.shape[0] * scale)
        new_w = round(img.shape[1] * scale)
        pad_h = (self.intput_spec[2] - new_h) / 2
        pad_w = (self.intput_spec[3] - new_w) / 2
        top = round(pad_h - 0.1)
        left = round(pad_w - 0.1)
        bottom = round(pad_h + 0.1)
        right = round(pad_w + 0.1)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return img

    def postprocess(self, result: np.ndarray, shapes: np.ndarray) -> list:
        gain = 640.0 / shapes
        gain = np.min(gain, axis=1)
        pad_h = np.round(self.intput_spec[2] - shapes[:, 0] * gain) / 2 - 0.1
        pad_w = np.round(self.intput_spec[3] - shapes[:, 1] * gain) / 2 - 0.1

        # reshape for broadcasting
        gain = gain.reshape(len(shapes), 1, 1)
        pad_h = pad_h.reshape(len(shapes), 1, 1)
        pad_w = pad_w.reshape(len(shapes), 1, 1)

        p = np.empty_like(result)
        dw = result[:, 2, :] / 2
        dh = result[:, 3, :] / 2
        p[:, 0, :] = result[:, 0, :] - dw
        p[:, 1, :] = result[:, 1, :] - dh
        p[:, 2, :] = result[:, 0, :] + dw
        p[:, 3, :] = result[:, 1, :] + dh
        p[:, 4, :] = result[:, 4, :]

        p[:, [0, 2], :] -= pad_w
        p[:, [1, 3], :] -= pad_h
        p[:, [0, 1, 2, 3], :] /= gain
        p[:, [0, 2], :] = np.clip(
            p[:, [0, 2], :], 0, shapes[:, 1].reshape(len(shapes), 1, 1)
        )
        p[:, [1, 3], :] = np.clip(
            p[:, [1, 3], :], 0, shapes[:, 0].reshape(len(shapes), 1, 1)
        )

        results = [[] for _ in range(len(p))]
        for i, r in enumerate(p):
            filtered = r[:, r[4] > self.confidence_threshold]
            boxes = np.round(np.moveaxis(filtered[:4], 0, 1)).astype(np.uint32)
            confs = filtered[4]
            idxs = cv2.dnn.NMSBoxes(
                boxes, confs, self.confidence_threshold, self.nms_threshold  # type: ignore
            )
            for idx in idxs:
                results[i].append(
                    [boxes[idx, 0], boxes[idx, 1], boxes[idx, 2], boxes[idx, 3]]
                )
        return results
