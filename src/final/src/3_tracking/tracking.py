#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tracking Node:
Subscribes : /target_position  (std_msgs/Float32MultiArray
Publishes  : /motion_control/cmd_vel (geometry_msgs/Twist)
"""

import math
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


class SimpleFollowerPD(object):
    def __init__(self):
        rospy.init_node('simple_person_follower_pd')

        # Parameters

        # Vision geometry
        self.img_w = float(rospy.get_param('~image_width2', 640))
        self.img_h = float(rospy.get_param('~image_height2', 480))
        self.cx_ref = self.img_w / 2.0

        # Controller gains
        self.kp_ang = rospy.get_param('~kp_angular2', 1.3)
        self.kd_ang = rospy.get_param('~kd_angular2', 0)
        self.max_ang = rospy.get_param('~max_angular2', 1.0)

        # Linear controller
        self.kp_lin = rospy.get_param('~kp_linear2', 1.5)
        self.max_lin_fwd  = rospy.get_param('~max_linear_forward2', 1.0)
        self.max_lin_back = rospy.get_param('~max_linear_backward2', -0.6)
        self.target_h  = rospy.get_param('~target_box_height_px2', 400)
        self.tolerance = rospy.get_param('~size_tolerance_px2', 25)

        # Smoother heading‑weight
        self.eps_heading = rospy.get_param('~epsilon_heading2', 0.8)

        # Driving aids
        self.lost_frame_thr = rospy.get_param('~lost_frame_threshold2', 5)
        self.min_ang_effort = rospy.get_param('~min_angular_effort2', 0.35)  # rad/s
        self.creep_fwd      = rospy.get_param('~creep_forward_speed2', 0.3)  # m/s, 0 = disabled

        self.cmd_pub = rospy.Publisher('/person_tracking/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/target_position', Float32MultiArray, self.bbox_cb, queue_size=1)

        # State variables
        self.prev_err = 0.0
        self.prev_t   = rospy.Time.now()
        self.last_twist = Twist()
        self.lost_count = 0

        rospy.on_shutdown(self._stop_robot)
        rospy.loginfo("SimpleFollowerPD ready.")
        rospy.spin()

    def bbox_cb(self, msg):
        x, y, w, h = msg.data
        rospy.loginfo("bbox  x=%d  y=%d  w=%d  h=%d", x, y, w, h)

        # lost‑frame handling
        if w == 0:
            self.lost_count += 1
            if self.lost_count < self.lost_frame_thr:
                self.cmd_pub.publish(self.last_twist)
            else:
                self._stop_robot()
            return
        else:
            self.lost_count = 0

        now = rospy.Time.now()
        dt  = (now - self.prev_t).to_sec() or 1e-6

        err_x  = (x + w / 2.0 - self.cx_ref) / self.cx_ref
        derr_x = (err_x - self.prev_err) / dt

        ang = self.kp_ang * err_x + self.kd_ang * derr_x
        ang = max(-self.max_ang, min(self.max_ang, ang))

        # Ensure enough torque to overcome static friction
        if 0.0 < abs(ang) < self.min_ang_effort:
            ang = math.copysign(self.min_ang_effort, ang)

        self.prev_err = err_x
        self.prev_t   = now

        # distance 
        size_err = h - self.target_h
        if abs(size_err) <= self.tolerance:
            lin = 0.0
        else:
            lin = self.kp_lin * (-size_err / self.target_h)
            lin = max(self.max_lin_back, min(self.max_lin_fwd, lin))

        # speed
        heading_weight = max(0.0, 1.0 - (err_x / self.eps_heading) ** 2)
        lin *= heading_weight

        if abs(ang) >= self.min_ang_effort and lin == 0.0 and self.creep_fwd > 0.0:
            lin = min(self.creep_fwd, self.max_lin_fwd)

        # publish
        twist = Twist()
        twist.linear.x  = lin
        twist.angular.z = ang
        self.cmd_pub.publish(twist)

        self.last_twist = twist
        rospy.loginfo("cmd   linear %.3f m/s | angular %.3f rad/s", lin, ang)

    def _stop_robot(self):
        self.last_twist = Twist()
        self.cmd_pub.publish(self.last_twist)
        rospy.loginfo("cmd   linear 0.000 m/s | angular 0.000 rad/s")

if __name__ == '__main__':
    try:
        SimpleFollowerPD()
    except rospy.ROSInterruptException:
        pass
