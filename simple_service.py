#!/usr/bin/env python3
import time
from scipy.spatial.transform import Rotation as Rlib
from enum import Enum
import numpy as np
import time
import cv2
import os

import rospy
from cv_bridge import CvBridge

from typing import Callable, List, Dict, Union
from wrappers.interface import (
    DetectorWrapper,
    MatcherWithoutDetectorWrapper,
    MatcherWrapper,
)
from wrappers.bf import BfMatcher
from wrappers.superglue import SuperGlueMatcher
from wrappers.superpoint import SuperPointDetector
from utils.utils import get_camera_matrix, white_balance

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


class MatcherNode:
    """
    ROSWrapper to offer matching services
    """

    def __init__(
        self,
        is_debug=False,
        node_name="matcher",
        detector: Union[DetectorWrapper, None] = None,
        matcher: Union[MatcherWrapper, MatcherWithoutDetectorWrapper, None] = None,
    ):
        rospy.init_node(
            node_name, anonymous=True, log_level=rospy.DEBUG if is_debug else rospy.INFO
        )
        self.services: List[rospy.Service] = []
        self.bridge = CvBridge()

        rospy.logdebug("Starting matcher and detector setup...")
        self.matcher = matcher if matcher is not None else SuperGlueMatcher()

        self.use_detector = not isinstance(self.matcher, MatcherWithoutDetectorWrapper)
        if self.use_detector:
            self.detector = detector if detector is not None else SuperPointDetector()
            rospy.logwarn(
                "Provided matcher requires a keypoint detector, but no detector passed... \
                Using SuperPoint as default detector"
            )
        else:
            self.detector = None
            if detector is not None:
                rospy.logwarn(
                    "Provided matcher does not require a keypoint detector, but a detector was passed... \
                    Ignoring the detector"
                )
        rospy.logdebug("Matcher and detector setup complete")

        rospy.logdebug("Setting up buffer...")
        self.buffer = []
        self.idx = 0

        rospy.logdebug("Setting up path...")
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.debug = is_debug
        if is_debug:
            self.debug_path = os.path.join(self.path, "debug")
            if not os.path.isdir(self.debug_path):
                os.mkdir(self.debug_path)

        rospy.logdebug("Setting template folder...")
        template_path = os.path.join(self.path, "templates")
        self.available_templates = [
            os.path.join(template_path, x) for x in os.listdir(template_path)
        ]

        rospy.logwarn("Warming up models")
        start = rospy.Time.now()
        self.warmup()
        end = rospy.Time.now()
        rospy.logdebug("Warmup complete in {} seconds".format((end - start).to_sec()))

        rospy.logdebug("Setting up service...")
        self.offer_services()

        rospy.logwarn("Serices offered!")

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
            rospy.logwarn("More than 1 item in buffer. Picking first 2...")

        img1 = self.buffer[0]
        img2 = self.buffer[1]

        if self.use_detector:
            # Get Keypoints
            start = time.time()
            keypoints = self.detector.pairwise(img1, img2)
            end = time.time()
            rospy.logdebug(f"Time taken for detector: {end - start}")

            # Get Matches
            start = time.time()
            matches = self.matcher(keypoints, num_keypoints)
            end = time.time()
            rospy.logdebug(f"Time taken for matcher: {end - start}")

        else:
            start = time.time()
            matches = self.matcher(img1, img2, num_keypoints)
            end = time.time()
            rospy.logdebug(f"Time taken for matcher: {end - start}")

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
            topic_name, CompressedImage, timeout=1
        )  # type: ignore

        cv2_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(
            img,
        )

        if self.debug:
            cv2.imwrite(
                os.path.join(self.debug_path, f"current_{len(self.buffer) % 2}.jpg"),
                cv2_img,
            )
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

        rospy.loginfo(f"Num. Keypoints: {num_keypoints}")
        rospy.loginfo(f"Matching with template: {template_name}")

        relevant_templates = sorted(
            list(filter(lambda x: template_name in x, self.available_templates))
        )
        if num_keypoints <= 0 or len(relevant_templates) <= 0:
            resp = MatchToTemplateResponse()
            resp.result = 1
            return resp

        relevant_template_path = relevant_templates[0]
        rospy.logdebug(f"Using {relevant_template_path}")
        cv2_img: np.ndarray = cv2.imread(relevant_template_path)
        cv2_img = white_balance(cv2_img)
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_path, "template.jpg"), cv2_img)

        self._add_img(cv2_img)

        results = self._infer_matches(num_keypoints)

        if self.debug:
            kp1 = results["ref_keypoints"]  # Camera image
            kp2 = results["cur_keypoints"]  # Template

            pts1 = np.int32(np.array(kp1))
            pts2 = np.int32(np.array(kp2))

            rospy.logwarn(
                "Essential Matrix --- Note that E matrix algorithm in OpenCV cannot be used for planar cases"
            )
            # Refer to https://www.robots.ox.ac.uk/~vgg/publications/1998/Torr98c/torr98c.pdf 
            # for the different degenerate cases. In particular, case of dim(N) = 3
            E, inliers = cv2.findEssentialMat(
                pts1,
                pts2,
                focal=1.0,
                pp=(0.0, 0.0),
                method=cv2.USAC_MAGSAC,
                prob=0.999,
                threshold=5.0,
            )
            inliers = inliers > 0
            print(E)
            _, R, T, _ = cv2.recoverPose(
                E, pts2, pts1
            )  # Mapping from camera image to template
            print(
                "RPY : (deg.)",
                -1
                * Rlib.from_matrix(R).as_euler(
                    "zyx", degrees=True
                ),  # Yaw Pitch Roll in Camera Frame which is RDF -> Roll Pitch Yaw
            )
            print("Translation", T)

            rospy.logwarn("Homography --- For planar scenes ")

            M, inliers = cv2.findHomography(
                pts1, pts2, cv2.USAC_MAGSAC, 5.0
            )  # Mapping from camera image to template
            inliers = inliers > 0
            print(M)

            img1 = cv2.imread(os.path.join(self.debug_path, "current_0.jpg"))
            img2 = cv2.imread(os.path.join(self.debug_path, "template.jpg"))

            h, w, _ = img2.shape
            pts = np.float32([[1, 1], [w - 1, 1], [w - 1, h - 1], [1, h - 1]]).reshape(
                -1, 1, 2
            )
            dst = cv2.perspectiveTransform(
                pts, np.linalg.inv(M)
            )  # Inverse because we want template -> camera mapping
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
                        "RPY : (deg.)",
                        -1 * Rlib.from_matrix(Rs[i]).as_euler("zyx", degrees=True),
                    )
                    print("Translation", Ts[i][:, 0])
                    print("Normal", Ns[i][:, 0])

            print()

        return self._to_correct_format(results, lambda: MatchToTemplateResponse())  # type: ignore

    def warmup(self):
        debug_state = self.debug
        self.debug = False

        # Constants for warmup
        num_keypoints = 1000
        width = 720
        height = 640
        channel = 3

        if self.use_detector:
            for _ in range(10):
                cv2_img: np.ndarray = np.random.rand(width, height, channel).astype(
                    np.float32
                )
                cv2_img = white_balance(cv2_img)
                self._add_img(cv2_img)

                cv2_img: np.ndarray = np.random.rand(width, height, channel).astype(
                    np.float32
                )
                cv2_img = white_balance(cv2_img)
                self._add_img(cv2_img)

                self._infer_matches(num_keypoints)
        else:
            for _ in range(10):
                img1: np.ndarray = np.random.rand(width, height, channel).astype(
                    np.float32
                )
                img2: np.ndarray = np.random.rand(width, height, channel).astype(
                    np.float32
                )

                self.matcher(img1, img2, num_keypoints)

        self.debug = debug_state


class Mode(Enum):
    BRUTE_FORCE = 0
    SUPERGLUE = 1
    COARSE_LOFTR = 2
    LOFTR = 3


if __name__ == "__main__":
    # Configuration
    DEBUG_MODE = True
    detector_config = {
        "cuda": True,
    }
    matcher_mode: Mode = Mode.SUPERGLUE
    matcher_config = {
        "cuda": True,
        "weights": (
            "outdoor" if matcher_mode != Mode.COARSE_LOFTR else "LoFTR_teacher"
        ),
    }

    # Setup matcher
    matcher = None
    if matcher_mode == Mode.SUPERGLUE:
        matcher = SuperGlueMatcher(matcher_config)
    elif matcher_mode == Mode.BRUTE_FORCE:
        matcher = BfMatcher(matcher_config)
    elif matcher_mode == Mode.COARSE_LOFTR:
        matcher = CoarseLoftrMatcher(matcher_config)
    elif matcher_mode == Mode.LOFTR:
        matcher = LoftrMatcher(matcher_config)
    else:
        raise ValueError("Invalid matcher mode")

    # Setup detector
    detector = None
    if isinstance(matcher, MatcherWrapper):
        detector = SuperPointDetector(detector_config)

    matcher_node = MatcherNode(
        DEBUG_MODE,
        detector=detector,
        matcher=matcher,
    )

    rospy.spin()
