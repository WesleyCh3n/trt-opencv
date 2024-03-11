import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from trt.engine import TensorRTInfer
from trt.utils import batch


class R50Model:
    def __init__(self, model_path: str, batch_size: int):
        self.bs = batch_size
        self.model = TensorRTInfer(model_path, batch_size)

    def __call__(self, imgs: list) -> np.ndarray:
        blobs = np.zeros((len(imgs), 3, 112, 112), dtype=np.float32)

        num_workers = os.cpu_count()
        if num_workers is None:
            raise RuntimeError("Unable to determine number of CPU cores")
        executor = ThreadPoolExecutor(int(num_workers / 2))
        jobs = [executor.submit(self.preprocess, img, i) for i, img in enumerate(imgs)]
        for job in as_completed(jobs):
            blob, i = job.result()
            blobs[i] = blob

        result = np.zeros((len(imgs), 512), dtype=np.float32)
        for i, b in enumerate(batch(blobs, self.bs)):
            result[self.bs * i : self.bs * (i + 1)] = self.model.infer(b, len(b))

        return result

    def preprocess(self, image_path: str, index: int):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0  # transform.ToTensor()
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))  # to CxWxH
        blob = np.expand_dims(img, axis=0)  # Add batch dimension
        blob = np.ascontiguousarray(blob, np.float32)
        return blob, index
