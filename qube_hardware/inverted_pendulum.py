from QUBE import QUBE
from control import COM_PORT
from time import sleep, time
import numpy as np

# Initialize QUBE
port = COM_PORT
baudrate = 115200
qube = QUBE(port, baudrate)

# Physical parameters
m_p = 0.1  # Pendulum stick mass (kg)
l = 0.095  # Pendulum length, pivot to tip (m)
l_com = l / 2  # Distance to COM (m)
J = (1 / 3) * m_p * l * l  # Inertia (kg*m^2)
g = 9.81  # Gravitational constant (m/s^2)

# Swing-up parameters
Er = 0.015  # Reference energy (J)
ke = 30  # Swing-up gain
u_max = 1.8  # Saturation
balance_range = 35.0  # Switch to balance inside this range (deg)
handoff_range = 65.0  # Fade swing-up effort inside this range (deg)
SWINGUP_DIRECTION = 1

# Balance parameters
s = 0.33
kp_theta = 2 * s
kd_theta = 0.125 * s
kp_pos = 0.07 * s
kd_pos = 0.06 * s

# Filter parameters
twopi = 3.141592 * 2
wc = 500 / twopi
wc2 = 500 / twopi
wc3 = 500 / twopi
y_k_last = 0
y2_k_last = 0
y3_k_last = 0

# Python/serial cannot reliably match the 2000 Hz Teensy loop, so this is a
# conservative host-side rate. Velocity estimates use the measured loop time.
freq = 200
target_dt = 1.0 / freq

# Program variables
prevAngle = 0
prevPos = 0
last = time()
t_balance = time()
t_reset = time()
mode = 0
lastMode = 0
reset = False


def constrain(value, lower, upper):
    return max(lower, min(upper, value))


def low_pass(previous, value, dt, cutoff):
    alpha = constrain(cutoff * dt, 0, 1)
    return previous + alpha * (value - previous)


def pendulum_angle_top_zero():
    angle = qube.getPendulumAngle()
    angle = ((angle + 180) % 360) - 180
    return angle - 180 if angle > 0 else angle + 180


def setup():
    qube.setRGB(999, 999, 999)
    qube.resetMotorEncoder()
    sleep(1)
    qube.resetPendulumEncoder()
    qube.setMotorVoltage(0)
    qube.update()
    sleep(1)


def swingup(angle, dt):
    global prevAngle, y3_k_last

    angularV_deg = (angle - prevAngle) / dt
    prevAngle = angle

    angle_rad = np.deg2rad(angle)
    angularV_rad = np.deg2rad(angularV_deg)

    E = 0.5 * J * angularV_rad * angularV_rad + m_p * g * l_com * (
        1 - np.cos(angle_rad)
    )
    u = SWINGUP_DIRECTION * ke * (E - Er) * (-angularV_rad * np.cos(angle_rad))
    handoff_scale = constrain(abs(angle) / handoff_range, 0.15, 1.0)
    u *= handoff_scale
    u_sat = min(u_max, max(-u_max, u))

    voltage = u_sat * (8.4 * 0.095 * 0.085) / 0.042
    y3_k = low_pass(y3_k_last, voltage, dt, wc3)
    y3_k_last = y3_k

    qube.setMotorVoltage(y3_k)


def settle_motor(position, dt):
    global prevPos, y2_k_last

    v = (position - prevPos) / dt
    prevPos = position
    y2_k = low_pass(y2_k_last, v, dt, wc2)
    y2_k_last = y2_k

    u_pos = kp_pos * 3 * position + kd_pos * y2_k
    qube.setMotorVoltage(-u_pos)


def balance(position, angle, dt):
    global prevAngle, prevPos, y_k_last, y2_k_last

    u_dot = (angle - prevAngle) / dt
    v = (position - prevPos) / dt

    y_k = low_pass(y_k_last, u_dot, dt, wc)
    y2_k = low_pass(y2_k_last, v, dt, wc2)

    u_ang = kp_theta * angle + kd_theta * y_k
    u_pos = kp_pos * position + kd_pos * y2_k
    u = u_pos + u_ang

    qube.setMotorVoltage(u)
    prevAngle = angle
    prevPos = position
    y_k_last = y_k
    y2_k_last = y2_k


def loop():
    global last, mode, t_balance, reset, t_reset, lastMode

    now = time()
    dt = now - last
    if dt < target_dt:
        return
    last = now

    position = qube.getMotorAngle()
    angle = pendulum_angle_top_zero()

    if mode == 0 and -balance_range < angle < balance_range:
        mode = 1
        t_balance = now
    if mode == 1 and not (-balance_range < angle < balance_range):
        mode = 0
        if now - t_balance > 1:
            reset = True
            t_reset = now

    if reset:
        while time() - t_reset < 2:
            settle_motor(position, target_dt)
            qube.update()
            position = qube.getMotorAngle()
            sleep(target_dt)

        reset = False
        return

    if mode == 0:
        swingup(angle, dt)
    if mode == 1:
        balance(position, angle, dt)

    qube.update()
    lastMode = mode


if __name__ == "__main__":
    try:
        setup()
        while True:
            loop()
    except KeyboardInterrupt:
        print("Interrupted. Stopping motor.")
    finally:
        qube.setMotorVoltage(0)
        qube.update()
