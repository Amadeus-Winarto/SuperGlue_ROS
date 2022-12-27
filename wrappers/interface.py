from typing import Dict
import numpy as np


class MatcherWrapper:
    def __init__(self, config=None, *args, **kwargs) -> None:
        self.config = config

    def __call__(self, kptsdescs: Dict, num_keypoints: int) -> Dict:
        raise NotImplementedError


class DetectorWrapper:
    def __init__(self, config=None, *args, **kwargs) -> None:
        self.config = config

    def __call__(self, img: np.ndarray) -> Dict:
        raise NotImplementedError


class MatcherWithoutDetectorWrapper:
    def __init__(self, config=None, *args, **kwargs) -> None:
        self.config = config

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        raise NotImplementedError
