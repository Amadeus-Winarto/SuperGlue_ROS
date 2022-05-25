#! /usr/bin/env python3
#
# THX https://github.com/introlab/rtabmap

import random
import numpy as np
import torch

import os

path = (os.path.dirname(os.path.abspath(__file__))).replace(
    "/tools", ""
) + "/models/SuperPointPretrainedNetwork/"

from demo_superpoint import SuperPointFrontend

torch.set_grad_enabled(False)

device = "cpu"
superpoint = []


def init(cuda):
    # print("SuperPoint python init()")

    global device
    device = "cuda" if torch.cuda.is_available() and cuda else "cpu"

    # This class runs the SuperPoint network and processes its outputs.
    global superpoint
    superpoint = SuperPointFrontend(
        weights_path=path + "superpoint_v1.pth",
        nms_dist=4,
        conf_thresh=0.015,
        nn_thresh=1,
        cuda=cuda,
    )


def detect(imageBuffer):
    # print("SuperPoint python detect()")
    global device
    image = np.asarray(imageBuffer)
    image = image.astype("float32") / 255.0

    global superpoint
    pts, desc, heatmap = superpoint.run(image)
    # return float: Kpts:Nx3, Desc:NxDim
    # use copy to make sure memory is correctly re-ordered
    pts = np.float32(np.transpose(pts)).copy()
    desc = np.float32(np.transpose(desc)).copy()

    return pts, desc


if __name__ == "__main__":
    # test
    init(True)
    detect(np.random.rand(640, 480) * 255)
