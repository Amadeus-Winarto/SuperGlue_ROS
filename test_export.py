from models.SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from models.SuperGluePretrainedNetwork.models.superglue import SuperGlue
import os

import torch
import torch.onnx

superpoint_config = {
    "descriptor_dim": 256,
    "nms_radius": 4,
    "keypoint_threshold": 0.005,
    "max_keypoints": -1,
    "remove_borders": 4,
    "path": os.path.abspath(os.path.join(__file__, "../.."))
    + "/models/SuperGluePretrainedNetworks/models/weights/superpoint_v1.pth",
    "cuda": True,
}
ts_superpoint = SuperPoint(superpoint_config).eval().cuda()

superglue_config = {
    "descriptor_dim": 256,
    "weights": "indoor",
    "keypoint_encoder": [32, 64, 128, 256],
    "GNN_layers": ["self", "cross"] * 9,
    "sinkhorn_iterations": 100,
    "match_threshold": 0.2,
    "cuda": True,
}
ts_superglue = SuperGlue(superglue_config).eval().cuda()
print("Loaded")

torch.jit.save(
    ts_superpoint,
    "/home/amadeus/bbauv/src/SuperGlue_ROS/models/SuperGluePretrainedNetwork/models/weights/superpoint_v1.zip",
)
torch.jit.save(
    ts_superglue,
    f"/home/amadeus/bbauv/src/SuperGlue_ROS/models/SuperGluePretrainedNetwork/models/weights/superglue_{superglue_config['weights']}.zip",
)
