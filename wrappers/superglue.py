import logging
import os

import torch

from models.SuperGluePretrainedNetwork.models.superglue import SuperGlue

from typing import Dict


class SuperGlueMatcher:
    default_config = {
        "descriptor_dim": 256,
        "weights": "outdoor",
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
        "cuda": True,
    }

    def __init__(self, config=None) -> None:
        if config is None:
            config = {}
        self.config = self.default_config
        self.config = {**self.config, **config}

        logging.info("SuperGlue matcher config: ")
        logging.info(self.config)

        self.device = (
            "cuda" if torch.cuda.is_available() and self.config["cuda"] else "cpu"
        )

        assert self.config["weights"] in ["indoor", "outdoor"]
        path_ = os.path.abspath(
            os.path.join(__file__, "../..")
        ) + "/models/SuperGluePretrainedNetwork/models/weights/superglue_{}.pth".format(
            self.config["weights"]
        )
        self.config["path"] = path_

        parent_dir = os.path.dirname(path_)
        ref_file = os.path.basename(path_).split(".")[0]
        ts_file = os.path.join(parent_dir, ref_file + ".zip")

        logging.info("Creating SuperGlue matcher...")
        if False: # os.path.isfile(ts_file):
            self.superglue = torch.jit.load(ts_file).eval().to(self.device)
        else:
            self.superglue = SuperGlue(self.config).eval().to(self.device)

    def preprocess(self, kptdescs: Dict) -> Dict:
        logging.info("Prepare input data for superglue...")
        data = {}
        data["image_size0"] = (
            torch.from_numpy(kptdescs["ref"]["image_size"]).float().to(self.device)
        )
        data["image_size1"] = (
            torch.from_numpy(kptdescs["cur"]["image_size"]).float().to(self.device)
        )

        # print(data["image_size0"])
        # raise NotImplementedError

        if "torch" in kptdescs["cur"]:
            data["scores0"] = kptdescs["ref"]["torch"]["scores"][0].unsqueeze(0)
            data["keypoints0"] = kptdescs["ref"]["torch"]["keypoints"][0].unsqueeze(0)
            data["descriptors0"] = kptdescs["ref"]["torch"]["descriptors"][0].unsqueeze(
                0
            )

            data["scores1"] = kptdescs["cur"]["torch"]["scores"][0].unsqueeze(0)
            data["keypoints1"] = kptdescs["cur"]["torch"]["keypoints"][0].unsqueeze(0)
            data["descriptors1"] = kptdescs["cur"]["torch"]["descriptors"][0].unsqueeze(
                0
            )
        else:
            data["scores0"] = (
                torch.from_numpy(kptdescs["ref"]["scores"])
                .float()
                .to(self.device)
                .unsqueeze(0)
            )
            data["keypoints0"] = (
                torch.from_numpy(kptdescs["ref"]["keypoints"])
                .float()
                .to(self.device)
                .unsqueeze(0)
            )
            data["descriptors0"] = (
                torch.from_numpy(kptdescs["ref"]["descriptors"])
                .float()
                .to(self.device)
                .unsqueeze(0)
                .transpose(1, 2)
            )

            data["scores1"] = (
                torch.from_numpy(kptdescs["cur"]["scores"])
                .float()
                .to(self.device)
                .unsqueeze(0)
            )
            data["keypoints1"] = (
                torch.from_numpy(kptdescs["cur"]["keypoints"])
                .float()
                .to(self.device)
                .unsqueeze(0)
            )
            data["descriptors1"] = (
                torch.from_numpy(kptdescs["cur"]["descriptors"])
                .float()
                .to(self.device)
                .unsqueeze(0)
                .transpose(1, 2)
            )

        return data

    def forward(self, data) -> Dict:
        logging.info("Matching keypoints with superglue...")
        with torch.no_grad():
            pred: Dict = self.superglue(data)
        return pred

    def postprocess(self, pred: Dict, kptdescs: Dict, num_keypoints: int) -> Dict:
        # get matching keypoints
        kpts0 = kptdescs["ref"]["keypoints"]
        kpts1 = kptdescs["cur"]["keypoints"]

        matches = pred["matches0"][0].cpu().numpy()
        confidence = pred["matching_scores0"][0].cpu().detach().numpy()

        # Sort them in the order of their confidence.
        match_conf = []
        for i, (m, c) in enumerate(zip(matches, confidence)):
            match_conf.append([i, m, c])
        match_conf = sorted(match_conf, key=lambda x: -x[2])

        valid = [[x[0], x[1]] for x in match_conf if x[1] > -1]
        v0 = [x[0] for x in valid]
        v1 = [x[1] for x in valid]
        mkpts0 = kpts0[v0]
        mkpts1 = kpts1[v1]

        ret_dict = {
            "ref_keypoints": mkpts0[:num_keypoints],
            "cur_keypoints": mkpts1[:num_keypoints],
            "match_score": confidence[v0][:num_keypoints],
        }

        return ret_dict

    def __call__(self, kptdescs: Dict, num_keypoints: int) -> Dict:
        preprocessed = self.preprocess(kptdescs)
        preds = self.forward(preprocessed)
        return self.postprocess(preds, kptdescs, num_keypoints)
