import logging
from typing import Dict

import torch
import numpy as np

import kornia as K
import kornia.feature as KF


from .interface import MatcherWithoutDetectorWrapper


class LoftrMatcher(MatcherWithoutDetectorWrapper):
    def __init__(self, config=None) -> None:
        if config is None:
            config = {}
        self.config = {**config}

        logging.debug("Loftr config: ")
        logging.debug(self.config)

        self.device = (
            "cuda" if torch.cuda.is_available() and self.config["cuda"] else "cpu"
        )

        assert self.config["weights"] in ["indoor", "outdoor"]
        self.matcher = (
            KF.LoFTR(pretrained=self.config["weights"]).eval().to(self.device)
        )

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = K.image_to_tensor(img, False).float() / 255.0
        img = K.color.bgr_to_grayscale(img.to(self.device))
        return img

    def __call__(self, img1: np.ndarray, img2: np.ndarray, num_keypoints: int) -> Dict:
        img1 = self.preprocess(img1)
        img2 = self.preprocess(img2)

        input_dict = {
            "image0": img1,
            "image1": img2,
        }

        with torch.inference_mode():
            correspondences = self.matcher(input_dict)

        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        mconf = correspondences["confidence"].cpu().numpy()

        # filter only the most confident features
        if num_keypoints is not None and num_keypoints < mkpts0.shape[0]:
            n_top = num_keypoints
            indices = np.argsort(mconf)[::-1]
            indices = indices[:n_top]
            mkpts0 = mkpts0[indices, :]
            mkpts1 = mkpts1[indices, :]

        return {
            "ref_keypoints": mkpts0[:num_keypoints],
            "cur_keypoints": mkpts1[:num_keypoints],
            "match_score": mconf[:num_keypoints],
        }
