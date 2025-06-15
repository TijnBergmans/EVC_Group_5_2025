#!/bin/bash

# setup ros environment
catkin_make
source "devel/setup.bash"
roslaunch final S5.launch