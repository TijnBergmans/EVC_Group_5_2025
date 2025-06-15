#!/bin/bash

# setup ros environment
catkin_make
source "devel/setup.bash"
roslaunch final pipeline_display.launch