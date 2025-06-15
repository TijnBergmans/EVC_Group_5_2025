#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float64MultiArray
import time
from math import pi
import sys
from encoderDriver import *

# Publishing speed
SPEED_PUBLISHING_RATE = 50


class MotorCalibrationNode(object):
    def __init__(self):
        rospy.init_node("Encoder Processor")
        rospy.loginfo("[Encoder Processor] Initializing…")

        # Pins for encoders
        GPIO_MOTOR_ENCODER_1 = 18
        GPIO_MOTOR_ENCODER_2 = 19

        # Wheel‑encoder driver instances
        self.driver_1 = WheelEncoderDriver(GPIO_MOTOR_ENCODER_1)
        self.driver_2 = WheelEncoderDriver(GPIO_MOTOR_ENCODER_2)

        # Constants
        self.increment_radians = 2 * pi / 137  # radians per encoder tick
        self.dt = 1.0 / SPEED_PUBLISHING_RATE

        # State trackers
        self.left_prev_ticks = 0
        self.right_prev_ticks = 0
        self.cumulative_left = 0.0
        self.cumulative_right = 0.0

        # Publishers
        self.speed_pub = rospy.Publisher(
            "/speed", Float64MultiArray, queue_size=1
        )
        self.radians_pub = rospy.Publisher(
            "/radians_turned", Float64MultiArray, queue_size=1
        )

        self.timer = rospy.Timer(
            rospy.Duration(self.dt), self.publish_measurements, oneshot=False
        )

        rospy.loginfo("[encoder_calibration_node] Ready.")

    def ticks_to_radians(self, ticks):
        """Convert encoder ticks to physical wheel radians."""
        return ticks * self.increment_radians

    def publish_measurements(self, _event):
        # Current tick count from both wheels
        left_ticks = self.driver_1._ticks
        right_ticks = self.driver_2._ticks

        # Compute tick increment since last callback
        left_inc_ticks = left_ticks - self.left_prev_ticks
        right_inc_ticks = right_ticks - self.right_prev_ticks

        # Save for next cycle
        self.left_prev_ticks = left_ticks
        self.right_prev_ticks = right_ticks

        # Convert to radian increment
        left_inc_rad = self.ticks_to_radians(left_inc_ticks)
        right_inc_rad = self.ticks_to_radians(right_inc_ticks)

        # Update cumulative radians
        self.cumulative_left += left_inc_rad
        self.cumulative_right += right_inc_rad

        # Compute angular speed
        left_speed = left_inc_rad / self.dt
        right_speed = right_inc_rad / self.dt

        # Publish speed
        speed_msg = Float64MultiArray()
        speed_msg.data = [left_speed, right_speed]
        self.speed_pub.publish(speed_msg)

        # Publish radians
        rad_msg = Float64MultiArray()
        rad_msg.data = [self.cumulative_left, self.cumulative_right]
        self.radians_pub.publish(rad_msg)

    def shutdown(self):
        rospy.loginfo("[encoder_calibration_node] Shutting down.")
        self.driver_1.close()
        self.driver_2.close()

if __name__ == "__main__":
    try:
        node = MotorCalibrationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.shutdown()
