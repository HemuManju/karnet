import os

from bisect import bisect_left

import numpy as np

import rosbag
from cv_bridge import CvBridge
import cv2


class CvBridgeError(TypeError):
    """
    This is the error raised by :class:`cv_bridge.CvBridge` methods when they fail.
    """

    pass


def read_compressed_ros_image(cmprs_img_msg, desired_encoding=None):

    str_msg = cmprs_img_msg.data
    buf = np.ndarray(shape=(1, len(str_msg)), dtype=np.uint8, buffer=cmprs_img_msg.data)
    im = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)

    if desired_encoding == "passthrough":
        return im

    try:
        res = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except RuntimeError as e:
        raise CvBridgeError(e)

    return res


def get_nonexistant_shard_path(fname_path):
    """
    Get the path to a filename which does not exist by incrementing path.

    Examples
    --------
    >>> get_nonexistant_path('/etc/issue')
    '/etc/issue-1'
    >>> get_nonexistant_path('whatever/1337bla.py')
    'whatever/1337bla.py'
    """
    shard_start = 0

    while True:
        if not os.path.exists(fname_path % shard_start):
            return fname_path, shard_start
        else:
            shard_start += 1


def get_closest_index(query, targets):
    """Retrieves the index of the element in targets that is closest to query O(log n)
    Args:
        query (float): query value
        targets (list): Sorted list of float values
    Returns:
        idx (int): index of the closest element in the array to x
    """
    idx = bisect_left(targets, query)
    if idx >= len(targets):
        idx = len(targets) - 1
    d = abs(targets[idx] - query)

    # check if index above or below is closer to query
    if targets[idx] < query and idx < len(targets) - 1:
        if abs(targets[idx + 1] - query) < d:
            return idx + 1
    elif targets[idx] > query and idx > 0:
        if abs(targets[idx - 1] - query) < d:
            return idx - 1
    return idx
