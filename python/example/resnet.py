import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from trt.resnet import R50Model

np.set_printoptions(suppress=True)

model = R50Model("../../model/old/512-model-fp16.trt", 256, True)
r = model(["../../20240114-041024.091_00_126_6_0.jpg"])
print(r[0][:10])
