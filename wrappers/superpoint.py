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
        + "/models/SuperPointPretrainedNetwork/superpoint_v1.pth",
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

        logging.info("Creating SuperPoint detector...")
        self.superpoint = SuperPoint(self.config).eval().to(self.device)

    def __call__(self, image) -> Dict:
        try:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logging.error(e)
            pass  # Squeezed gray image array

        logging.debug("detecting keypoints with superpoint...")
        image_tensor = image2tensor(image, self.device)

        with torch.no_grad():
            pred = self.superpoint({"image": image_tensor})

        ret_dict = {
            "image_size": np.array([image.shape[0], image.shape[1]]),
            "torch": pred,
            "keypoints": pred["keypoints"][0].cpu().detach().numpy(),
            "scores": pred["scores"][0].cpu().detach().numpy(),
            "descriptors": pred["descriptors"][0].cpu().detach().numpy().transpose(),
        }

        return ret_dict
