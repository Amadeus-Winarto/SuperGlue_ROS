#!/usr/bin/env python3
from scipy.spatial.transform import Rotation as Rlib
from threading import Thread, Lock
import numpy as np
import logging
import copy
import cv2
import os

import rospy
from cv_bridge import CvBridge

from typing import Callable, List, Dict, Union
from wrappers.bf import BfMatcher
from wrappers.superglue import SuperGlueMatcher
from wrappers.superpoint import SuperPointDetector

from sensor_msgs.msg import CompressedImage
from superglue_ros.msg import Keypoint, KeypointsDict
from superglue_ros.srv import RegisterImage, RegisterImageRequest, RegisterImageResponse
from superglue_ros.srv import ClearBuffer, ClearBufferRequest, ClearBufferResponse
from superglue_ros.srv import MatchImages, MatchImagesRequest, MatchImagesResponse
from superglue_ros.srv import (
    MatchToTemplate,
    MatchToTemplateRequest,
    MatchToTemplateResponse,
)


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - (
        (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result[:, :, 2] = result[:, :, 2] - (
        (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def get_camera_matrix():
    SIM_K = np.array(
        [
            [407.0646129842357, 0.0, 384.5],
            [0.0, 407.0646129842357, 246.5],
            [0.0, 0.0, 1.0],
        ]
    )

    REAL_K = np.array(
        [
            [436.40875244140625, 0.0, 510.88065980075044],
            [0.0, 467.6256103515625, 376.3738157469634],
            [0.0, 0.0, 1.0],
        ]
    )

    return SIM_K


class MatcherNode:
    """
    ROSWrapper to offer matching services
    """

    def __init__(
        self,
        is_debug=False,
        use_superglue=True,
        node_name="matcher",
        detector_config: Union[Dict, None] = None,
        matcher_config: Union[Dict, None] = None,
    ):
        rospy.init_node(node_name)
        self.services: List[rospy.Service] = []
        self.bridge = CvBridge()

        self.detector_config = detector_config
        self.matcher_config = matcher_config

        self.detector = SuperPointDetector(self.detector_config)

        if use_superglue:
            self.matcher = SuperGlueMatcher(self.matcher_config)
        else:
            self.matcher = BfMatcher(self.matcher_config)

        self.buffer = []
        self.idx = 0

        self.path = os.path.dirname(os.path.abspath(__file__))
        self.debug = is_debug
        if is_debug:
            self.debug_path = os.path.join(self.path, "debug")
            if not os.path.isdir(self.debug_path):
                os.mkdir(self.debug_path)

        template_path = os.path.join(self.path, "templates")
        self.available_templates = [
            os.path.join(template_path, x) for x in os.listdir(template_path)
        ]

        self.offer_services()

    def offer_services(self):
        # Add the services here
        self.services.append(
            rospy.Service("registerImage", RegisterImage, self.register_img)
        )
        self.services.append(
            rospy.Service("clearBuffer", ClearBuffer, self.clear_buffer)
        )
        self.services.append(
            rospy.Service("matchImages", MatchImages, self.match_images)
        )
        self.services.append(
            rospy.Service("matchToTemplate", MatchToTemplate, self.match_to_template)
        )

    def _add_img(self, img: np.ndarray) -> RegisterImageResponse:
        response = RegisterImageResponse()
        if len(self.buffer) < 2:
            self.idx += 1
            self.buffer.append(img)
            response.result = 0  # No issues
        else:
            response.result = 1  # Buffer overflow
        return response

    def _infer_matches(self, num_keypoints: int):
        if len(self.buffer) < 2:
            raise ValueError("No matching possible!")
        elif len(self.buffer) > 2:
            logging.warning("More than 1 item in buffer. Picking first 2...")

        img1 = self.buffer[0]
        img2 = self.buffer[1]

        # Get Keypoints
        keypoints = self.detector.pairwise(img1, img2)

        # Get Matches
        matches = self.matcher(keypoints, num_keypoints)

        self.buffer = []

        if self.debug:
            from tools.tools import plot_matches

            img = plot_matches(
                cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
                matches["ref_keypoints"],
                matches["cur_keypoints"],
                matches["match_score"],
                layout="lr",
            )
            cv2.imwrite(os.path.join(self.debug_path, "test.jpg"), img)
        return matches

    def _to_correct_format(
        self,
        results: Dict,
        response_creator: Callable[
            [], Union[MatchImagesResponse, MatchToTemplateResponse]
        ],
    ) -> Union[MatchImagesResponse, MatchToTemplateResponse]:
        response = response_creator()
        response.keypoints_dict = KeypointsDict()

        response.keypoints_dict.ref_keypoints = []
        response.keypoints_dict.cur_keypoints = []

        for k in results["ref_keypoints"]:
            kp = Keypoint()
            kp.coord = k
            response.keypoints_dict.ref_keypoints.append(kp)

        for k in results["cur_keypoints"]:
            kp = Keypoint()
            kp.coord = k
            response.keypoints_dict.cur_keypoints.append(kp)
        response.keypoints_dict.match_score = results["match_score"]  # List of floats
        return response

    def register_img(self, req: RegisterImageRequest) -> RegisterImageResponse:
        topic_name = req.topic_name
        img: CompressedImage = rospy.wait_for_message(
            topic_name, CompressedImage, timeout=2
        )
        cv2_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(img)
        cv2_img = white_balance(cv2_img)
        cv2.imwrite(os.path.join(self.debug_path, "current.jpg"), cv2_img)
        return self._add_img(cv2_img)

    def clear_buffer(self, _: ClearBufferRequest) -> ClearBufferResponse:
        response = ClearBufferResponse()
        if len(self.buffer) < 0:
            response.result = 1  # Buffer underflow
        else:
            self.buffer.clear()
            response.result = 0
        return response

    def match_images(self, request: MatchImagesRequest) -> MatchImagesResponse:
        num_keypoints = request.numKeypoints
        if num_keypoints <= 0:
            resp = MatchImagesResponse()
            resp.result = 1
            return resp

        results = self._infer_matches(num_keypoints)
        return self._to_correct_format(results, lambda: MatchImagesResponse())  # type: ignore

    def match_to_template(
        self, request: MatchToTemplateRequest
    ) -> MatchToTemplateResponse:
        num_keypoints = request.numKeypoints
        template_name = request.template_name
        if template_name == "":
            raise ValueError("template_name is empty!")

        print(num_keypoints)
        print(template_name)

        relevant_templates = sorted(
            list(filter(lambda x: template_name in x, self.available_templates))
        )
        if num_keypoints <= 0 or len(relevant_templates) <= 0:
            resp = MatchToTemplateResponse()
            resp.result = 1
            return resp

        relevant_template_path = relevant_templates[0]
        print(f"Using {relevant_template_path}")
        cv2_img: np.ndarray = cv2.imread(relevant_template_path)
        cv2_img = white_balance(cv2_img)
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_path, "template.jpg"), cv2_img)

        self._add_img(cv2_img)

        results = self._infer_matches(num_keypoints)

        if self.debug:
            from tools.tools import plot_matches

            img1 = cv2.imread(os.path.join(self.debug_path, "current.jpg"))
            img2 = cv2.imread(os.path.join(self.debug_path, "template.jpg"))

            img = plot_matches(
                cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
                results["ref_keypoints"][0:200],
                results["cur_keypoints"][0:200],
                results["match_score"][0:200],
                layout="lr",
            )

            cv2.imwrite(os.path.join(self.debug_path, "debug_matches_py.jpg"), img)

            # Do Homography here
            print("Homography")
            kp1 = results["ref_keypoints"]
            kp2 = results["cur_keypoints"]

            pts1 = np.int32(np.array(kp1))
            pts2 = np.int32(np.array(kp2))

            M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
            print(M)

            h, w, _ = img2.shape
            pts = np.float32([[1, 1], [w - 1, 1], [w - 1, h - 1], [1, h - 1]]).reshape(
                -1, 1, 2
            )
            dst = cv2.perspectiveTransform(pts, M)
            img_lines = cv2.polylines(img1, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            cv2.imwrite(
                os.path.join(self.debug_path, "debug_keypoints_py.jpg"), img_lines
            )

            _, Rs, Ts, Ns = cv2.decomposeHomographyMat(
                np.linalg.inv(M), get_camera_matrix()
            )

            for i in range(len(Rs)):

                if Ns[i].T.dot(np.array([[0], [0], [1]]))[0][0] < 0:
                    print(i)
                    print(
                        "YPR : (deg.)",
                        -1 * Rlib.from_matrix(Rs[i]).as_euler("yxz", degrees=True),
                    )
                    print("Translation", Ts[i][:, 0])
                    print("Normal", Ns[i][:, 0])

            print()

        return self._to_correct_format(results, lambda: MatchToTemplateResponse())  # type: ignore


if __name__ == "__main__":
    DEBUG_MODE = True
    matcher_node = MatcherNode(DEBUG_MODE)
    print("RUNNING")
    rospy.spin()
