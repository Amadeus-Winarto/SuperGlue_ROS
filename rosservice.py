from threading import Thread, Lock
import numpy as np
import logging
import copy
import cv2
import os

import rospy
from cv_bridge import CvBridge

from typing import Callable, List, Dict, Union
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
class ImageWrapper:
    def __init__(self, id: int, img: np.ndarray) -> None:
        self.id = id
        self.img = img
        self.thread = None
        self.results = None

    def register_thread(self, thread: Thread):
        self.thread = thread

    def add_results(self, results: Dict):
        self.results = results


class Buffer:
    def __init__(self) -> None:
        self.buffer: List[ImageWrapper] = []
        self.lock = Lock()

    def append(self, x) -> None:
        self.lock.acquire()
        self.buffer.append(x)
        self.lock.release()

    def clear(self) -> None:
        self.lock.acquire()
        self.buffer = []
        self.lock.release()

    def __len__(self) -> int:
        self.lock.acquire()
        length = len(self.buffer)
        self.lock.release()
        return length


class MatcherNode:
    """
    ROSWrapper to offer matching services
    """

    def __init__(
        self,
        node_name="superglue_matcher",
        detector_config: Union[Dict, None] = None,
        matcher_config: Union[Dict, None] = None,
    ):
        rospy.init_node(node_name)
        self.services: List[rospy.Service] = []
        self.bridge = CvBridge()

        self.detector_config = detector_config
        self.matcher_config = matcher_config

        self.detector = SuperPointDetector(self.detector_config)
        self.matcher = SuperGlueMatcher(self.matcher_config)

        self.buffer = Buffer()
        self.lock = Lock()
        self.idx = 0

        self.debug = True
        self.available_templates = [
            os.path.join("templates", x) for x in os.listdir("templates")
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

    def _get_keypoints(self, id: int):
        self.buffer.lock.acquire()
        relevant = [x for x in self.buffer.buffer if x.id == id]
        if len(relevant) != 1:
            self.buffer.lock.release()
            raise ValueError("Impossible!")
        else:
            img = copy.deepcopy(relevant[0].img)
            self.buffer.lock.release()

        kp = self.detector(img)

        self.buffer.lock.acquire()
        for x in self.buffer.buffer:
            if x.id == id:
                x.add_results(kp)
        self.buffer.lock.release()

    def _add_img(self, img: np.ndarray) -> RegisterImageResponse:
        response = RegisterImageResponse()
        if len(self.buffer) < 2:
            self.idx += 1
            content = ImageWrapper(self.idx, img)
            self.buffer.append(content)
            response.result = 0  # No issues

            # Get keypoints
            t = Thread(target=self._get_keypoints, args=(self.idx,))
            t.start()
            content.register_thread(t)
        else:
            response.result = 1  # Buffer overflow
        return response

    def _infer_matches(self, num_keypoints: int):
        if len(self.buffer) < 2:
            raise ValueError("No matching possible!")
        elif len(self.buffer) > 2:
            logging.warning("More than 1 item in buffer. Picking first 2...")

        self.buffer.lock.acquire()
        content1 = self.buffer.buffer[0]
        content2 = self.buffer.buffer[1]
        self.buffer.lock.release()

        content1.thread.join()  # type: ignore
        content2.thread.join()  # type: ignore

        keypoints = {"ref": content1.results, "cur": content2.results}
        matches = self.matcher(keypoints, num_keypoints)

        self.buffer.lock.acquire()
        content1 = self.buffer.buffer.pop(0)
        content2 = self.buffer.buffer.pop(0)
        self.buffer.lock.release()

        if self.debug:
            img1 = content1.img
            img2 = content2.img

            from tools.tools import plot_matches

            img = plot_matches(
                cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
                matches["ref_keypoints"],
                matches["cur_keypoints"],
                matches["match_score"],
                layout="lr",
            )
            cv2.imwrite("test.jpg", img)
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
        cv2.imwrite("/home/amadeus/bbauv/src/stereo/imgs/current.jpg", cv2_img)
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
        cv2.imwrite("/home/amadeus/bbauv/src/stereo/imgs/template.jpg", cv2_img)

        self._add_img(cv2_img)

        results = self._infer_matches(num_keypoints)
        return self._to_correct_format(results, lambda: MatchToTemplateResponse())  # type: ignore

    @staticmethod
    def threaded_test():
        import cv2

        matcher_node = MatcherNode()
        img1 = cv2.imread("/home/amadeus/bbauv/src/stereo/imgs/current.jpg")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        resp1 = matcher_node._add_img(img1)
        print("Result1 : ", resp1.result)

        img2 = cv2.imread("/home/amadeus/bbauv/src/stereo/imgs/template.jpg")
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        matcher_node._add_img(img2)

        assert len(matcher_node.buffer) == 2

        matches = matcher_node._infer_matches(500)

        from tools.tools import plot_matches

        img = plot_matches(
            cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
            matches["ref_keypoints"][0:200],
            matches["cur_keypoints"][0:200],
            matches["match_score"][0:200],
            layout="lr",
        )

        cv2.imshow("MATCHES", img)
        cv2.waitKey(0)

    @staticmethod
    def threaded_test2():
        import cv2

        matcher_node = MatcherNode()
        img1 = cv2.imread("/home/amadeus/bbauv/src/stereo/imgs/left_sim.jpg")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        resp1 = matcher_node._add_img(img1)
        print("Result1 : ", resp1.result)

        assert len(matcher_node.buffer) == 1

        img2 = cv2.imread(
            "/home/amadeus/bbauv/src/SuperGlue_ROS/templates/template.jpeg"
        )

        sample_req = MatchToTemplateRequest()
        sample_req.template_name = "template"
        sample_req.numKeypoints = 500

        resp = matcher_node.match_to_template(sample_req)

        from tools.tools import plot_matches

        img = plot_matches(
            cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
            [[x.coord[0], x.coord[1]] for x in resp.keypoints_dict.ref_keypoints],
            [[x.coord[0], x.coord[1]] for x in resp.keypoints_dict.cur_keypoints],
            resp.keypoints_dict.match_score,
            layout="lr",
        )

        cv2.imshow("MATCHES", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    # MatcherNode.threaded_test2()
    matcher_node = MatcherNode(
        detector_config={"cuda": False}, matcher_config={"cuda": False}
    )
    print("RUNNING")
    rospy.spin()
