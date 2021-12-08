#!/usr/bin/env python

# System.
import numpy as np
import time

# Third-party.
# import cv2

# ROS.
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

# Local package.
from shimizu_auto_color_calib.msg import colorir
from postprocess import process

DEFAUL_NODE_NAME = 'color_ir_backend_node'

pub = None

def callback(msg):
    start = time.time()
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', type(msg))

    def Image2Array(msg):
        data = np.fromstring(msg.data, dtype=np.uint8)
        data = data.reshape(msg.height, msg.width, -1)
        if data.shape[-1] == 1:
            data = data.reshape(data.shape[:2])
        return data

    raw = Image2Array(msg.raw)
    block = Image2Array(msg.block)
    point = Image2Array(msg.point)

    para, show = process(raw.astype(np.float32) / 255, block.astype(np.float32) / 255, point.astype(np.float32) / 255)

    show_msg = Image()
    show_msg.data = (show.clip(0, 1) * 255).astype(np.uint8).tostring()
    show_msg.width = show.shape[1]
    show_msg.height = show.shape[0]
    show_msg.step = show.shape[1] * 3
    show_msg.encoding = "bgr8"
    pub.publish(show_msg)

    print("Backend returns in %.3f s" % (time.time() - start))
    print('BGR_coeff = {}, vgn_alhpa = {}, len = {}'.format(
        para['coeff'], para['alpha'], para['length']))


def listener():
    global DEFAUL_NODE_NAME, pub

    rospy.init_node(DEFAUL_NODE_NAME, anonymous=True)

    rospy.loginfo('%s created. ' % (rospy.get_name()))

    pub = rospy.Publisher('color_ir_backend', Image, queue_size=1)

    rospy.Subscriber('/color_ir_inference', colorir, callback)

    rospy.spin()


if __name__ == '__main__':
    # import sys
    # print("sys.version =", sys.version)
    listener()
