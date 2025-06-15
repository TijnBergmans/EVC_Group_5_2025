#!/bin/bash

# setup ros environment
catkin_make
source "devel/setup.bash"
roslaunch final S4.launch