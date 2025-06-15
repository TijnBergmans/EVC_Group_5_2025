#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Range
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from collections import deque

class ObstacleAvoidanceNode(object):
    def __init__(self):
        rospy.init_node('obstacle_avoidance_node', anonymous=True)
        rospy.loginfo("Starting Obstacle Avoidance Node with Encoders…")

        # PARAMETERS
        self.threshold = rospy.get_param('threshold', 0.5)
        self.forward_speed = rospy.get_param('forward_speed', 0.3)
        self.turn_speed = rospy.get_param('turn_speed', 5)
        self.window_size = rospy.get_param('window_size', 3)

        # DISTANCES (in radians)
        self.turn_left_target = rospy.get_param('turn_left_target', 5)
        self.forward_target = rospy.get_param('forward_target', 15.0)
        self.turn_right_target = rospy.get_param('turn_right_target', 3.4)

        # PUBLISHER
        self.cmd_pub = rospy.Publisher('/obstacle_avoidance/cmd_vel', Twist, queue_size=1)

        # SUBSCRIBERS
        rospy.Subscriber('/tof/depth', Range, self.range_cb, queue_size=1)
        rospy.Subscriber('/radians_turned', Float64MultiArray, self.radians_cb)

        rospy.on_shutdown(self.stop_robot)

        # STATE VARIABLES
        self.state = 'IDLE'
        self.recent_ranges = deque(maxlen=self.window_size)

        self.left_encoder = 0.0
        self.right_encoder = 0.0
        self.maneuver_start_left = 0.0
        self.maneuver_start_right = 0.0

    def range_cb(self, msg):
        if self.state != 'IDLE':
            return
        self.recent_ranges.append(msg.range)
        rospy.loginfo("TOF range reading: %.3f m", msg.range)

        if len(self.recent_ranges) < self.window_size:
            return

        if sum(self.recent_ranges)/len(self.recent_ranges) < self.threshold:
            self.start_backward()
            self.recent_ranges.clear()

    def radians_cb(self, msg):
        # Update latest encoder values from the message
        self.left_encoder = msg.data[0]
        self.right_encoder = msg.data[1]
        self.update_maneuver_progress()

    def start_turn_left(self):
        rospy.loginfo("Starting turn left")
        self.state = 'TURN_LEFT'
        self.maneuver_start_left = self.left_encoder
        self.maneuver_start_right = self.right_encoder

        cmd = Twist()
        cmd.angular.z = self.turn_speed
        self.cmd_pub.publish(cmd)

    def start_forward(self):
        rospy.loginfo("Starting forward")
        self.state = 'FORWARD'
        self.maneuver_start_left = self.left_encoder
        self.maneuver_start_right = self.right_encoder

        cmd = Twist()
        cmd.linear.x = self.forward_speed
        self.cmd_pub.publish(cmd)

    def start_backward(self):
        rospy.loginfo("Starting backward")
        self.state = 'BACKWARD'
        self.maneuver_start_left = self.left_encoder
        self.maneuver_start_right = self.right_encoder

        cmd = Twist()
        cmd.linear.x = -1 * self.forward_speed
        self.cmd_pub.publish(cmd)

    def start_turn_right(self):
        rospy.loginfo("Starting turn right")
        self.state = 'TURN_RIGHT'
        self.maneuver_start_left = self.left_encoder
        self.maneuver_start_right = self.right_encoder

        cmd = Twist()
        cmd.angular.z = -self.turn_speed
        self.cmd_pub.publish(cmd)

    def update_maneuver_progress(self):

        if self.state == 'BACKWARD':
            delta_left = abs(self.left_encoder - self.maneuver_start_left)
            delta_right = abs(self.right_encoder - self.maneuver_start_right)
            avg_delta = (delta_left + delta_right) / 2.0
            if avg_delta >= self.forward_target:
                self.start_turn_left()

        elif self.state == 'TURN_LEFT':
            delta_left = abs(self.left_encoder - self.maneuver_start_left)
            delta_right = abs(self.right_encoder - self.maneuver_start_right)
            rospy.loginfo("TURN_LEFT: Δleft=%.3f, Δright=%.3f, target=%.3f",
                          delta_left, delta_right, self.turn_left_target)
            if max(delta_left, delta_right) >= self.turn_left_target:
                self.start_forward()

        elif self.state == 'FORWARD':
            delta_left = abs(self.left_encoder - self.maneuver_start_left)
            delta_right = abs(self.right_encoder - self.maneuver_start_right)
            avg_delta = (delta_left + delta_right) / 2.0
            if avg_delta >= self.forward_target:
                self.start_turn_right()

        elif self.state == 'TURN_RIGHT':
            delta_left = abs(self.left_encoder - self.maneuver_start_left)
            delta_right = abs(self.right_encoder - self.maneuver_start_right)
            if max(delta_left, delta_right) >= self.turn_right_target:
                self.finish_maneuver()

    def finish_maneuver(self):
        rospy.loginfo("Maneuver done. Stopping.")
        self.stop_robot()
        self.state = 'IDLE'

    def stop_robot(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ObstacleAvoidanceNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
