import logging
import os
from typing import Dict

import torch
import numpy as np

from models.Coarse_LoFTR_TRT.loftr import LoFTR
from models.Coarse_LoFTR_TRT.loftr.utils.cvpr_ds_config import default_cfg
from models.Coarse_LoFTR_TRT.utils import (
    make_query_image,
    get_coarse_match,
    make_student_config,
)
from .interface import MatcherWithoutDetectorWrapper


class CoarseLoftrMatcher(MatcherWithoutDetectorWrapper):
    default_config = default_cfg

    def __init__(self, config=None) -> None:
        if config is None:
            config = {}
        self.config = self.default_config
        self.config = {**self.config, **config}
        self.config["input_size"] = (
            self.config["input_width"],
            self.config["input_height"],
        )

        logging.debug("CoarseLoftr config: ")
        logging.debug(self.config)

        self.device = (
            "cuda" if torch.cuda.is_available() and self.config["cuda"] else "cpu"
        )
        self.matcher = LoFTR(make_student_config(self.config)).eval().to(self.device)

        self.config["path"] = os.path.abspath(
            os.path.join(
                __file__,
                "..",
                "..",
                "models",
                "Coarse_LoFTR_TRT",
                "weights",
                self.config["weights"] + ".pt",
            )
        )

        checkpoint = torch.load(self.config["path"], map_location=self.device)
        if checkpoint is not None:
            state_dict = checkpoint["model_state_dict"]
            self.matcher.load_state_dict(state_dict, strict=False)
            device = torch.device(self.device)
            self.matcher = self.matcher.eval().to(device=device)
            logging.debug("Successfully loaded pre-trained weights.")
        else:
            logging.error("Failed to load weights")

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = make_query_image(img, self.config["input_size"])
        return torch.from_numpy(img)[None][None].to(device=self.device) / 255.0

    def __call__(self, img1: np.ndarray, img2: np.ndarray, num_keypoints: int) -> Dict:
        img1 = self.preprocess(img1)
        img2 = self.preprocess(img2)

        with torch.no_grad():
            conf_matrix, _ = self.matcher(img1, img2)
        conf_matrix = conf_matrix.cpu().numpy()

        mkpts0, mkpts1, mconf = get_coarse_match(
            conf_matrix,
            self.config["input_size"][1],
            self.config["input_size"][0],
            self.config["resolution"][0],
        )

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
