#!/bin/bash

# setup ros environment
catkin_make
source "devel/setup.bash"
roslaunch ros_deep_learning detectnet.ros1.launch \
         model_name:=pednet \
         input:=/camera/image_undistorted
