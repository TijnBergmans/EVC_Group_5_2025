#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Search Node:
Publishes: /search_path/cmd_vel  (geometry_msgs/Twist)
"""
import rospy
from geometry_msgs.msg import Twist

class SearchPathNode(object):
    def __init__(self):
        rospy.init_node('search_path_node', anonymous=True)
        rospy.loginfo("Starting Search Path Node…")


        self.rate_hz       = 1
        self.forward_speed = 0.45
        self.angular_speed = 0.8

        # publisher
        self.cmd_pub = rospy.Publisher(
            '/search_path/cmd_vel', Twist, queue_size=1)

        # ensure we stop on shutdown
        rospy.on_shutdown(self.stop_robot)

        self.run()

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        cmd = Twist()

        # circle forward+rotate
        cmd.linear.x  = self.forward_speed
        cmd.angular.z = self.angular_speed

        while not rospy.is_shutdown():
            self.cmd_pub.publish(cmd)
            rate.sleep()

    def stop_robot(self):
        rospy.loginfo("Stopping Search Path Node…")
        self.cmd_pub.publish(Twist())

if __name__ == '__main__':
    try:
        SearchPathNode()
    except rospy.ROSInterruptException:
        pass
