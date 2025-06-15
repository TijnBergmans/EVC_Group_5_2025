#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Motor Node
Subscribes:   /motion_control/cmd_vel  (geometry_msgs/Twist)
Publishes:    (drives wheels via DaguWheelsDriver.set_wheels_speed)
"""
import os
import json
import math
import threading
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from motorDriver import DaguWheelsDriver

# Calibration file written by old motornode
CALIB_FILE        = os.path.expanduser("~/.jetbot_wheel_calibration.json")
LEFT_ORIENTATION  = -1
RIGHT_ORIENTATION = 1

# Robot geometry
WHEEL_AXIS_M   = 0.171   # distance between wheels [m]
WHEEL_RADIUS_M = 0.034   # wheel radius [m]

class MotorNode(object):
    def __init__(self):
        rospy.init_node('motor_node', anonymous=False)
        rospy.loginfo("Starting MotorNode…")

        # driver instance
        self.driver = DaguWheelsDriver()
        # calibration tables
        self._calib_rel   = None
        self._calib_left  = None
        self._calib_right = None
        self._load_calibration()
        self._rad_lock = threading.Lock()

        # subscribe to the final cmd_vel
        rospy.Subscriber(
            '/motion_control/cmd_vel',
            Twist,
            self.cmd_vel_cb,
            queue_size=1,
            buff_size=2**24)

        # ensure motor stop on shutdown
        rospy.on_shutdown(self._shutdown)

    def _load_calibration(self):
        """Load the PWM→rad/s tables from disk, if they exist."""
        if not os.path.isfile(CALIB_FILE):
            rospy.logwarn("No calibration file at %s; using default scaling", CALIB_FILE)
            return
        try:
            with open(CALIB_FILE) as fh:
                data = json.load(fh)
            self._calib_rel   = np.asarray(data["relative"],  dtype=np.float64)
            self._calib_left  = np.asarray(data["left_rps"],  dtype=np.float64)
            self._calib_right = np.asarray(data["right_rps"], dtype=np.float64)
            rospy.loginfo("Loaded wheel calibration (%d points)", len(self._calib_rel))
        except Exception as e:
            rospy.logerr("Failed to load calibration: %s", e)

    def _rad_to_rel(self, rad_per_s, wheel):
        if self._calib_rel is None: 
            print("cant find calibration")
            orient = LEFT_ORIENTATION if wheel == "left" else RIGHT_ORIENTATION
            scale  = max(min(rad_per_s / 10.0, 1.0), -1.0)
            return orient * scale

        table = self._calib_left  if wheel == "left"  else self._calib_right
        rels  = self._calib_rel

        if table[0] > table[-1]:
            table = table[::-1]
            rels  = rels[::-1]
        clamped = np.clip(rad_per_s, table.min(), table.max())
        return float(np.interp(clamped, table, rels))

    def cmd_vel_cb(self, msg):

        linear  = msg.linear.x
        angular = msg.angular.z

        left_rps  = (linear - angular * (WHEEL_AXIS_M/2.0)) / WHEEL_RADIUS_M
        right_rps = (linear + angular * (WHEEL_AXIS_M/2.0)) / WHEEL_RADIUS_M
        
        # map via calibration
        l_rel = self._rad_to_rel(left_rps,  "left")
        r_rel = self._rad_to_rel(right_rps, "right")
        
        # send to driver
        self.driver.set_wheels_speed(l_rel, r_rel)

    def _shutdown(self):

        rospy.loginfo("MotorNode shutdown: stopping motors.")
        self.driver.set_wheels_speed(0.0, 0.0)
        try:
            self.driver.close()
        except Exception:
            pass

if __name__ == "__main__":
    node = MotorNode()
    rospy.spin()