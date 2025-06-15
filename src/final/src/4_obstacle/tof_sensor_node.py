#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
4_obstacle/tof_sensor_node.py

S4a: TOF Sensor Reading
  • Publishes: /tof/depth   (sensor_msgs/Range)

Reads from the VL53L0X TOF via tofDriver.VL53L0X and emits a Range message.
"""
import rospy
from sensor_msgs.msg import Range
from tofDriver import VL53L0X

class ToFSensorNode(object):
    def __init__(self):
        rospy.init_node('tof_sensor_node', anonymous=True)
        rospy.loginfo("Starting TOF Sensor Node…")

        # parameters
        self.frame_id     = rospy.get_param('frame_id',    'tof_link')
        self.rate_hz      = rospy.get_param('rate_obj',        10)
        # VL53L0X has ~30 mm–2 m working range by default
        self.min_range    = rospy.get_param('min_range',   0.03)
        self.max_range    = rospy.get_param('max_range',   2.0)
        self.fov          = rospy.get_param('field_of_view', 0.015)  # ~15 mrad

        # publisher
        self.pub = rospy.Publisher(
            '/tof/depth', Range, queue_size=1)

        # instantiate sensor driver
        self.sensor = VL53L0X()

        # ensure sensor is closed on shutdown
        rospy.on_shutdown(self.shutdown)

        self.spin()

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        msg = Range()
        msg.header.frame_id    = self.frame_id
        msg.radiation_type     = Range.INFRARED
        msg.field_of_view      = self.fov
        msg.min_range          = self.min_range
        msg.max_range          = self.max_range

        while not rospy.is_shutdown():
            dist_mm = self.sensor.read_distance()
            if dist_mm is not None:
                # convert to meters
                msg.header.stamp = rospy.Time.now()
                msg.range        = dist_mm * 1e-3
                self.pub.publish(msg)
            else:
                rospy.logwarn_throttle(5, "TOF read returned None")
            rate.sleep()

    def shutdown(self):
        rospy.loginfo("Shutting down TOF Sensor Node, closing driver…")
        try:
            self.sensor.close()
        except Exception:
            pass

if __name__ == '__main__':
    try:
        ToFSensorNode()
    except rospy.ROSInterruptException:
        pass
