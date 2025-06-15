#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Subscribes:
  /camera/image_undistorted      (sensor_msgs/Image)
  /detectnet/detections          (vision_msgs/Detection2DArray)

Publishes:
  /obj_detect/person_detect      (final/DetectedPerson)
"""
import rospy, cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, RegionOfInterest
from vision_msgs.msg import Detection2DArray
from final.msg import DetectedPerson

class PersonCropper(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.last_image = None

        rospy.Subscriber("/camera/image_undistorted", Image,
                         self.img_cb, queue_size=1)
        rospy.Subscriber("/detectnet/detections", Detection2DArray,
                         self.det_cb, queue_size=1)

        self.pub = rospy.Publisher("/obj_detect/person_detect",
                                   DetectedPerson, queue_size=10)

    # cache the most recent frame (detectnet and camera are already time-aligned)
    def img_cb(self, msg):
        self.last_image = msg

    def det_cb(self, det_msg):
        if self.last_image is None:
            return

        cv = self.bridge.imgmsg_to_cv2(self.last_image, "bgr8")
        h, w = cv.shape[:2]

        for det in det_msg.detections:
            # assume first result entry is the top hypothesis
            if det.results[0].id != 0:           # COCO id 0 = person
                continue

            cx = det.bbox.center.x
            cy = det.bbox.center.y
            bw = det.bbox.size_x
            bh = det.bbox.size_y

            x1 = int(max(cx - bw / 2, 0))
            y1 = int(max(cy - bh / 2, 0))
            x2 = int(min(cx + bw / 2, w))
            y2 = int(min(cy + bh / 2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = cv[y1:y2, x1:x2]
            ok, buf = cv2.imencode(".jpg", crop)
            if not ok:
                continue

            # fill DetectedPerson
            out = DetectedPerson()
            out.image = CompressedImage()
            out.image.header.stamp = rospy.Time.now()
            out.image.format = "jpeg"
            out.image.data = np.array(buf).tobytes()

            out.bbox = RegionOfInterest()
            out.bbox.x_offset = x1
            out.bbox.y_offset = y1
            out.bbox.width  = x2 - x1
            out.bbox.height = y2 - y1

            self.pub.publish(out)

if __name__ == "__main__":
    rospy.init_node("person_cropper")
    PersonCropper()
    rospy.spin()
