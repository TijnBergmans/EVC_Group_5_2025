#!/bin/bash

# setup ros environment
catkin_make
source "devel/setup.bash"
roslaunch final tracking_test.launch