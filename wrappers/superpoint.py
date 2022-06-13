import logging
import os

import cv2
import torch
import numpy as np

from models.SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from tools.tools import image2tensor

from typing import Dict


class SuperPointDetector(object):
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "path": os.path.abspath(os.path.join(__file__, "../.."))
        + "/models/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth",
        "cuda": True,
    }

    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("SuperPoint detector config: ")
        logging.info(self.config)

        self.device = (
            "cuda" if torch.cuda.is_available() and self.config["cuda"] else "cpu"
        )
        path_ = self.config["path"]
        parent_dir = os.path.dirname(path_)
        ref_file = os.path.basename(path_).split(".")[0]
        ts_file = os.path.join(parent_dir, ref_file + ".zip")

        logging.info("Creating SuperPoint detector...")
        if os.path.isfile(ts_file):
            self.superpoint = torch.jit.load(ts_file).eval().to(self.device)
        else:
            self.superpoint = SuperPoint(self.config).eval().to(self.device)

    def preprocess(self, image) -> torch.Tensor:
        try:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logging.error(e)
            pass  # Squeezed gray image array

        logging.debug("detecting keypoints with superpoint...")
        image_tensor = image2tensor(image, self.device)
        return image_tensor

    def __call__(self, image) -> Dict:
        with torch.no_grad():
            pred = self.superpoint(self.preprocess(image))

        ret_dict = {
            "image_size": np.array([image.shape[0], image.shape[1]]),
            "torch": pred,
            "keypoints": pred["keypoints"][0].cpu().detach().numpy(),
            "scores": pred["scores"][0].cpu().detach().numpy(),
            "descriptors": pred["descriptors"][0].cpu().detach().numpy().transpose(),
        }

        return ret_dict

    def pairwise(self, image1, image2) -> Dict:
        image_tensor1 = self.preprocess(image1)
        image_tensor2 = self.preprocess(image2)

        with torch.no_grad():
            pred1 = self.superpoint(image_tensor1)  # type: ignore
            pred2 = self.superpoint(image_tensor2)  # type: ignore

        return {
            "ref": {
                "image_size": np.array([image1.shape[0], image1.shape[1]]),
                "torch": pred1,
            },
            "cur": {
                "image_size": np.array([image2.shape[0], image2.shape[1]]),
                "torch": pred2,
            },
        }
