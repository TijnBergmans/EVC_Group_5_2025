# Message to hold information about detected lines

std_msgs/Header header # Standard header for timestamp and frame ID

# Array of detected line segments [x1, y1, x2, y2]
int32[] line_segments
