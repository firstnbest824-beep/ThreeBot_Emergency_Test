std_msgs/Header header
string robot_id
string mode  # "IDLE" | "TELEOP" | "AUTO" | "ERROR"
bool estop_active
uint8 error_code
float32[5] position  # [X1, Y1, X2, Y2, Z]
float32[5] velocity  # [vX1, vY1, vX2, vY2, vZ]
float32[5] effort    # [Current/Torque]
