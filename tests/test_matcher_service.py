import unittest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from matcher_service import MatcherNode

from superglue_ros.srv import MatchToTemplateRequest


class TestMatcherNode(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.matcher_node = MatcherNode()

    def test_pipeline(self):
        import cv2

        img1 = cv2.imread("/home/amadeus/bbauv/src/stereo/imgs/current.jpg")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        resp1 = self.matcher_node._add_img(img1)
        self.assertEqual(resp1.result, 0)

        img2 = cv2.imread("/home/amadeus/bbauv/src/stereo/imgs/template.jpg")
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        resp2 = self.matcher_node._add_img(img2)
        self.assertEqual(resp2.result, 0)

        self.assertEqual(len(self.matcher_node.buffer), 2)

        matches = self.matcher_node._infer_matches(500)

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

    def test_pipeline2(self):
        import cv2

        img1 = cv2.imread("/home/amadeus/bbauv/src/stereo/imgs/current.jpg")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        resp1 = self.matcher_node._add_img(img1)
        print("Result1 : ", resp1.result)

        assert len(self.matcher_node.buffer) == 1

        img2 = cv2.imread("/home/amadeus/bbauv/src/stereo/imgs/template.jpg")

        sample_req = MatchToTemplateRequest()
        sample_req.template_name = "template"
        sample_req.numKeypoints = 500

        resp = self.matcher_node.match_to_template(sample_req)

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
    unittest.main()
