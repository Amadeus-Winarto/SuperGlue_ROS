import logging
import cv2

from typing import Dict, List, Tuple

import numpy as np

from .interface import MatcherWrapper


class BfMatcher(MatcherWrapper):
    def __init__(self, config=None) -> None:
        if config is not None:
            raise NotImplementedError

        self.matcher = cv2.BFMatcher(normType=cv2.NORM_L1)

    def preprocess(self, kptdescs: Dict) -> Dict:
        logging.info("Prepare input data for BF...")
        data = {}

        data["scores0"] = kptdescs["ref"]["scores"]
        data["scores1"] = kptdescs["cur"]["scores"]

        data["keypoints0"] = kptdescs["ref"]["keypoints"]
        data["keypoints1"] = kptdescs["cur"]["keypoints"]

        data["descriptors0"] = kptdescs["ref"]["descriptors"]
        data["descriptors1"] = kptdescs["cur"]["descriptors"]

        return data

    def forward(self, data) -> Dict:
        logging.info("Matching keypoints with superglue...")
        matches: List[Tuple[cv2.DMatch]] = self.matcher.knnMatch(
            data["descriptors0"], data["descriptors1"], k=2
        )

        # # Need to draw only good matches, so create a mask
        # matchesMask = [[0, 0] for _ in range(len(matches))]

        # # ratio test as per Lowe's paper
        # for i, (m, n) in enumerate(matches):  # type: ignore
        #     if m.distance < 0.8 * n.distance:
        #         matchesMask[i] = [1, 0]
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            # matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )

        img1 = cv2.imread("/home/amadeus/bbauv/src/stereo/imgs/current.jpg")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        img2 = cv2.imread("/home/amadeus/bbauv/src/stereo/imgs/template.jpg")
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img3 = cv2.drawMatchesKnn(
            img1,
            [cv2.KeyPoint(x[0], x[1], 1) for x in data["keypoints0"]],
            img2,
            [cv2.KeyPoint(x[0], x[1], 1) for x in data["keypoints1"]],
            matches,
            None,
            **draw_params
        )

        cv2.imshow("MATCHES", img3)
        cv2.waitKey(0)

        matches0 = []
        matches1 = []
        matching_scores0 = []
        matching_scores1 = []

        for x, y in matches:
            matches0.append(x.trainIdx)
            matches1.append(y.trainIdx)
            matching_scores0.append(x.distance)
            matching_scores1.append(y.distance)

        pred = {
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": matching_scores0,
            "matching_scores1": matching_scores1,
        }

        return pred

    def postprocess(self, pred: Dict, kptdescs: Dict, num_keypoints: int) -> Dict:
        # get matching keypoints
        kpts0 = kptdescs["ref"]["keypoints"]
        kpts1 = kptdescs["cur"]["keypoints"]

        matches = zip(
            pred["matches0"],
            pred["matches1"],
            pred["matching_scores0"],
            pred["matching_scores1"],
        )

        # Sort them in the order of their confidence.
        match_conf = []
        for m1, m2, e1, e2 in matches:
            c = 1 / (e1 + e2)
            match_conf.append([m1, m2, c])
        match_conf = sorted(match_conf, key=lambda x: x[2], reverse=True)

        valid = [[x[0], x[1]] for x in match_conf if x[1] > -1]
        v0 = [x[0] for x in valid]
        v1 = [x[1] for x in valid]
        mkpts0 = kpts0[v0]
        mkpts1 = kpts1[v1]

        ret_dict = {
            "ref_keypoints": np.array(mkpts0[:num_keypoints]),
            "cur_keypoints": np.array(mkpts1[:num_keypoints]),
            "match_score": np.array([x[2] for x in match_conf[:num_keypoints]]),
        }

        return ret_dict

    def __call__(self, kptdescs: Dict, num_keypoints: int) -> Dict:
        preprocessed = self.preprocess(kptdescs)
        preds = self.forward(preprocessed)
        return self.postprocess(preds, kptdescs, num_keypoints)
