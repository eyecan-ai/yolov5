

from models.abstract_yolo import AbstractYoloModel
import torchsummary
import torch
import numpy as np

model_cfg = '/home/daniele/work/workspace_python/yolov5/models/ayolo/yolov5x_abstract.yaml'
model = AbstractYoloModel(cfg=model_cfg).to('cpu')


#torchsummary.summary(model, (3, 256, 256), device='cpu')

x = torch.tensor(np.random.uniform(0, 1, (1, 3, 256, 256))).float()
outputs = model(x)

for out in outputs:
    print(out.shape)
