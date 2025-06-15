#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
5_state/state_manager_node.py

S5: State Manager and Motion Control
  • Subscribes to the three candidate cmd_vel topics
  • Publishes the “winning” Twist on /motion_control/cmd_vel
  • Priority (highest first): obstacle_avoidance, person_tracking, search_path
  - Motor functions can be paused by calling
    'rostopic pub /pause_toggle std_msgs/Bool "data: true"' in another terminal
"""
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool



class StateManagerNode(object):
    def __init__(self, node_name):
        rospy.init_node(node_name, anonymous=True)
        rospy.loginfo("Starting State Manager Node…")

        # priorities: 0=search, 1=tracking, 2=obstacle
        self.cmds = {
            'search':  Twist(),
            'track':   Twist(),
            'obstacle':Twist(),
        }
        self.mode = 0
        self.counter = 0
        self.paused = False

        # publish final cmd
        self.cmd_pub = rospy.Publisher(
            '/motion_control/cmd_vel', Twist, queue_size=1)

        # subscribe to all three sources
        rospy.Subscriber('/search_path/cmd_vel',
                         Twist, self.search_cb, queue_size=1)
        rospy.Subscriber('/person_tracking/cmd_vel',
                         Twist, self.track_cb,  queue_size=1)
        rospy.Subscriber('/obstacle_avoidance/cmd_vel',
                         Twist, self.obstacle_cb, queue_size=1)
        
        # For pausing the robot with spacebar
        self.pause_sub = rospy.Subscriber("/pause_toggle", Bool, self.pause_cb)

        # update/publish at fixed rate
        rate_hz = rospy.get_param('state_manager_rate', 10)
        self.timer = rospy.Timer(rospy.Duration(1.0/rate_hz),
                                 self.timer_cb)
        

        rospy.on_shutdown(self.shutdown)

    def search_cb(self, msg):
        self.cmds['search'] = msg

    def track_cb(self, msg):
        self.cmds['track'] = msg

    def obstacle_cb(self, msg):
        self.cmds['obstacle'] = msg

    def timer_cb(self, event):
        # choose highest-priority non-zero command
        out = Twist()

        if self.paused:
            # Robot is paused — send zero velocity
            self.cmd_pub.publish(out)
            return
    
        # obstacle avoidance wins if any component non-zero
        obs = self.cmds['obstacle']
        if obs.linear.x != 0.0 or obs.angular.z != 0.0:
            out = obs
            if self.mode!=1:
                rospy.loginfo("Switch to obstacle avoidance")
            self.mode = 1
        else:
            tr = self.cmds['track']
            if tr.linear.x != 0.0 or tr.angular.z != 0.0:
                out = tr
                self.counter = 0
                if self.mode!=2:
                    rospy.loginfo("Switch to Following")
                self.mode = 2
            else:
                self.counter += 1
                if(self.counter > 20):
                    #out = self.cmds['search']    
                    out = self.cmds['track']    
                    if self.mode!=3:
                        rospy.loginfo("Switch to Searching")
                    self.mode = 3          
        self.cmd_pub.publish(out)

    def pause_cb(self, msg):
        self.paused = msg.data
        rospy.loginfo("Paused: %s" % self.paused)


    def shutdown(self):
        rospy.loginfo("State Manager shutting down, stopping robot.")
        self.cmd_pub.publish(Twist())

if __name__ == '__main__':
    try:
        node = StateManagerNode('state_manager_node')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
