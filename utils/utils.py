import cv2
import numpy as np


def white_balance(img: np.ndarray):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - (
        (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1  # type: ignore
    )
    result[:, :, 2] = result[:, :, 2] - (
        (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1  # type: ignore
    )
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def get_camera_matrix(use_sim: bool = False):

    return (
        np.array(
            [
                [407.0646129842357, 0.0, 384.5],
                [0.0, 407.0646129842357, 246.5],
                [0.0, 0.0, 1.0],
            ]
        )
        if use_sim
        else np.array(
            [
                [436.40875244140625, 0.0, 510.88065980075044],
                [0.0, 467.6256103515625, 376.3738157469634],
                [0.0, 0.0, 1.0],
            ]
        )
    )
