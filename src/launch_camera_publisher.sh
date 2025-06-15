#!/bin/bash

# setup ros environment
catkin_make
source "devel/setup.bash"
roslaunch final camera_publisher_top.launch