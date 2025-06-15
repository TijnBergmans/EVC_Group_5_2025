#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

class CameraSubscriberNode:
    def __init__(self):
        rospy.loginfo("Initializing bb image subscriber node...")
        self.video_writer = None  # Will hold VideoWriter instance
        self.fps = 20.0  # Adjust FPS as appropriate

        self.sub_image = rospy.Subscriber(
            "/camera/image_bb",
            CompressedImage,
            self.image_cb,
            queue_size=1,
            buff_size=2**24
        )

        rospy.loginfo("bb image subscriber node initialized!")

    def image_cb(self, data):
        try:
            # Decode image
            image_np = np.frombuffer(data.data, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            # Initialize VideoWriter on first image (when frame size known)
            if self.video_writer is None:
                height, width, _ = image.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4
                self.video_writer = cv2.VideoWriter('output_video.mp4', fourcc, self.fps, (width, height))
                rospy.loginfo("VideoWriter initialized: {}x{}".format(width, height))

            # Write frame to video
            self.video_writer.write(image)

            # Display the image
            cv2.imshow("Bounding box view", image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr("Error processing CompressedImage: {}".format(e))

    def cleanup(self):
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node('bb_img_subscriber_node', anonymous=True)
    node = CameraSubscriberNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down image viewer.")
    finally:
        node.cleanup()
