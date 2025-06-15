cmake_minimum_required(VERSION 3.0.2)
project(final)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  DualCompressedImage.msg
  DetectedPerson.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs
)

include_directories(${catkin_INCLUDE_DIRS})

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## Install Python driver modules so imports work:
install(FILES
  src/4_obstacle/tofDriver.py
  src/5_state/motorDriver.py
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

## Install your node scripts
catkin_install_python(PROGRAMS
  src/1_vision_detection/camera_publisher_node.py
  src/1_vision_detection/bb_img_publisher_node.py
  src/1_vision_detection/bb_img_subscriber_node.py
  src/1_vision_detection/vision_detection_node.py

  src/2_identification/person_reid_node.py
  src/3_tracking/tracking.py
  src/4_obstacle/tof_sensor_node.py
  src/4_obstacle/obstacle_avoidance_node.py
  src/5_state/search_path_node.py
  src/5_state/state_manager_node.py
  src/5_state/motor_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
