#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
camera_publisher_node.py

Publishes: /camera/image_undistorted  (sensor_msgs/Image, encoding=BGR8)
"""
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraPublisherNode(object):
    def __init__(self):
        rospy.init_node('camera_publisher_node', anonymous=True)
        rospy.loginfo("Starting Camera Publisher Node…")

        # Publisher: raw (uncompressed) image
        self.pub = rospy.Publisher(
            '/camera/image_undistorted',
            Image,
            queue_size=1
        )
        self.bridge = CvBridge()

        # Parameters
        self.device_id = rospy.get_param('device_id')
        self.fps       = rospy.get_param('fps')
        self.width  = rospy.get_param('image_width')
        self.height = rospy.get_param('image_height')
        self.camera_matrix = np.array(rospy.get_param('~camera_matrix', []), dtype=np.float32).reshape((3, 3)) if rospy.has_param('~camera_matrix') else None
        self.dist_coeffs = np.array(rospy.get_param('~dist_coeffs', []), dtype=np.float32) if rospy.has_param('~dist_coeffs') else None

        # GStreamer pipeline for Jetson camera
        self.pipeline = self.gstreamer_pipeline()
        if self.pipeline is None:
            rospy.logerr("Pipeline could not be initialized!")
            rospy.signal_shutdown("Invalid camera pipeline")
            return

        # Initialise capture
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            rospy.logerr("Unable to open camera")
            rospy.signal_shutdown("Camera open failure")
            return

        rospy.loginfo("Camera publisher node initialised!")
        self.rate = rospy.Rate(self.fps)
        self.spin()

    def gstreamer_pipeline(self):
        if self.device_id == 0:
            return (
                'nvarguscamerasrc sensor-id=0 ! '
                'video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, '
                'format=(string)NV12, framerate=(fraction)%d/1 ! '
                'queue max-size-buffers=1 leaky=downstream ! '
                'nvvidconv ! video/x-raw, format=(string)BGRx ! '
                'videoconvert ! video/x-raw, format=(string)BGR ! '
                'queue max-size-buffers=1 leaky=downstream ! '
                'appsink drop=true sync=false' % (self.width, self.height, self.fps)
            )
        elif self.device_id == 1:
            return (
                'v4l2src device=/dev/video1 ! '
                'video/x-raw, width=(int)%d, height=(int)%d, '
                'format=(string)YUY2, framerate=(fraction)%d/1 ! '
                'videoconvert ! video/x-raw, format=(string)BGR ! '
                'appsink drop=true sync=false' % (self.width, self.height, self.fps)
            )
        else:
            return None

    def spin(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("Camera frame read failed")
                self.rate.sleep()
                continue

            # Optional undistortion
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

            # Convert to ROS Image message (BGR8)
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp = rospy.Time.now()

            # Publish
            self.pub.publish(img_msg)
            self.rate.sleep()

        self.cap.release()


if __name__ == '__main__':
    try:
        CameraPublisherNode()
    except rospy.ROSInterruptException:
        pass
