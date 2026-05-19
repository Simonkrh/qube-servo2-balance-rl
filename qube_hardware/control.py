from PID import *

COM_PORT = "auto"

# Using mac?
USING_MAC = False

# Target values
MOTOR_TARGET_ANGLE = 0  # degrees
PENDULUM_TARGET_ANGLE = 0  # degrees
MOTOR_TARGET_RPM = 0  # rpm (max 3500)

# Flip this to -1 if the motor moves away from the target instead of toward it.
MOTOR_CONTROL_DIRECTION = 1

pid = PID()


# Main function to control. Must return the voltage (control signal) to apply to the motor.
def control_system(dt, motor_angle, pendulum_angle, rpm):
    if dt <= 0:
        return 0

    return MOTOR_CONTROL_DIRECTION * pid.control(
        MOTOR_TARGET_ANGLE,
        motor_angle,
        dt,
    )


# This function is used to tune the PID controller with the GUI.
def setPidParams(_pid):
    pid.copy(_pid)  # Uncomment this line if you want to use the GUI to tune your PID live.
    return 0
