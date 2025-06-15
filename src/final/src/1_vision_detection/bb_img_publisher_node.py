#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Image Publisher
  • Publishes:  /camera/image_bb         (sensor_msgs/CompressedImage)
  • Subscribes: /camera/image_raw (sensor_msgs/CompressedImage)
                /obj_detect/bb_coord (sensor_msgs/RegionOfInterest)

"""
import time
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, RegionOfInterest
from final.msg import DetectedPerson

class ImagePublisherNode:
    def __init__(self, node_name):
        self.initialized = False
        rospy.loginfo("Initializing image publisher node...")
        self.cv_image = None
        self.latest_bb = None
        self.latest_bb_time = 0
        self.fps = 30

        self.node_name = node_name
        rospy.init_node(self.node_name, anonymous=True)

        self.sub_image = rospy.Subscriber(
            "/camera/image_undistorted",
            CompressedImage,
            self.image_cb,
            buff_size=2**24,
            queue_size=1
        )

        self.sub_bb = rospy.Subscriber(
            "/obj_detect/person_detect",
            DetectedPerson,
            self.bb_cb,
            buff_size=2**24,
            queue_size=1
        )

        self.first_image_received = False
        self.initialized = True
        rospy.loginfo("Image publisher node initialized!")

        self.image_pub = rospy.Publisher(
            '/camera/image_bb',
            CompressedImage,
            queue_size=1
        )

        rospy.Timer(rospy.Duration(0.1), self.timer_cb)
        
    def image_cb(self, data):
        try:
            self.cv_image = cv2.imdecode(np.frombuffer(data.data, np.uint8), cv2.IMREAD_COLOR)
            self.latest_stamp = data.header.stamp
        except Exception as e:
            rospy.logerr("Error decoding image: {}".format(e))

    def bb_cb(self, msg):
        self.latest_bb = msg.bbox
        self.latest_bb_time = time.time()

    def start_publishing(self):
        rate = rospy.Rate(self.fps)
        while not rospy.is_shutdown():
            if self.cv_image is None:
                rate.sleep()
                continue

            undistorted = self.cv_image.copy()

            image_with_bb = self.draw_bb(undistorted, self.latest_bb)

            # Send encoded image message
            encoded = cv2.imencode('.jpg', image_with_bb)[1].tobytes()

            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = encoded

            self.image_pub.publish(msg)
            rate.sleep()        
    
    def draw_bb(self, image, bb):
        if bb is None or bb.width == 0 or bb.height == 0:
            return image  # Nothing to draw

        x1 = bb.x_offset
        y1 = bb.y_offset
        x2 = x1 + bb.width
        y2 = y1 + bb.height

        # Draw bounding box
        image_bb = image.copy()
        cv2.rectangle(image_bb, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return image_bb
    
    def timer_cb(self, _):
        # If no new bb has been sent for 1 seconds, remove bb from image
        if time.time() - self.latest_bb_time > 1:
            self.latest_bb = RegionOfInterest(x_offset=0, y_offset=0, width=0, height=0)

if __name__ == "__main__":
    try:
        node = ImagePublisherNode("bb_img_publisher_node")
        node.start_publishing()
    except rospy.ROSInterruptException:
        pass