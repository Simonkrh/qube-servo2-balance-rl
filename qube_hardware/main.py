# ------------------------------------- AVAILABLE FUNCTIONS --------------------------------#
# qube.setRGB(r, g, b) - Sets the LED color of the QUBE. Color values range from [0, 999].
# qube.setMotorSpeed(speed) - Sets the motor speed. Speed ranges from [-999, 999].
# qube.setMotorVoltage(volts) - Applies the given voltage to the motor. Volts range from (-24, 24).
# qube.resetMotorEncoder() - Resets the motor encoder in the current position.
# qube.resetPendulumEncoder() - Resets the pendulum encoder in the current position.

# qube.getMotorAngle() - Returns the cumulative angular positon of the motor.
# qube.getPendulumAngle() - Returns the cumulative angular position of the pendulum.
# qube.getMotorRPM() - Returns the newest rpm reading of the motor.
# qube.getMotorCurrent() - Returns the newest reading of the motor's current.
# ------------------------------------- AVAILABLE FUNCTIONS --------------------------------#

from QUBE import *
from logger import *
from com import *
from liveplot import *
from control import *
from time import monotonic, sleep, time
import threading

# Replace with the Arduino port. Can be found in the Arduino IDE (Tools -> Port:)
port = COM_PORT
baudrate = 115200
qube = QUBE(port, baudrate)

CONTROL_FREQUENCY_HZ = 50
MAX_MOTOR_VOLTAGE = 1.5

# Resets the encoders in their current position.
qube.resetMotorEncoder()
qube.resetPendulumEncoder()
qube.setMotorVoltage(0)

# Enables logging - comment out to remove
enableLogging()

t_last = time()

motor_target = 0
pendulum_target = 0
rpm_target = 0
pid = PID()


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def stop_motor():
    try:
        qube.setMotorVoltage(0)
        qube.update()
    except Exception:
        pass


def control(data, lock, stop_event):
    global motor_target, pendulum_target, rpm_target, pid
    period = 1.0 / CONTROL_FREQUENCY_HZ
    next_tick = monotonic()

    try:
        while not stop_event.is_set():
            loop_started = monotonic()
            motor_target = MOTOR_TARGET_ANGLE
            pendulum_target = PENDULUM_TARGET_ANGLE
            rpm_target = MOTOR_TARGET_RPM

            # Updates the qube - Sends and receives data
            qube.update()
            qube.setRGB(0, 999, 0)

            # Gets the logdata and writes it to the log file
            logdata = qube.getLogData(motor_target, pendulum_target, rpm_target)
            save_data(logdata)

            # Multithreading stuff that must happen. Dont mind it.
            with lock:
                doMTStuff(data)

            # Get deltatime
            dt = getDT()

            # Set pid parameters using GUI
            setPidParams(pid)

            # Get states
            motor_degrees = qube.getMotorAngle()
            pendulum_degrees = qube.getPendulumAngle()
            rpm = qube.getMotorRPM()

            # Get control signal
            u = control_system(dt, motor_degrees, pendulum_degrees, rpm)
            u = clamp(u, -MAX_MOTOR_VOLTAGE, MAX_MOTOR_VOLTAGE)

            # Apply control signal
            qube.setMotorVoltage(u)

            next_tick += period
            sleep_time = next_tick - monotonic()
            if sleep_time > 0:
                sleep(sleep_time)
            else:
                next_tick = monotonic()
    except Exception as error:
        print("Control loop stopped:")
        print(error)
        stop_event.set()
    finally:
        stop_motor()


def getDT():
    global t_last
    t_now = time()
    dt = t_now - t_last
    t_last += dt
    return dt


def doMTStuff(data):
    packet = data[9]
    pid.copy(packet.pid)
    if packet.resetEncoders:
        qube.resetMotorEncoder()
        qube.resetPendulumEncoder()
        packet.resetEncoders = False

    new_data = qube.getPlotData(motor_target, pendulum_target, rpm_target)
    for i, item in enumerate(new_data):
        data[i].append(item)


if __name__ == "__main__":
    stop_event = threading.Event()
    try:
        _data = [[], [], [], [], [], [], [], [], [], Packet()]
        lock = threading.Lock()

        if not USING_MAC:
            thread2 = threading.Thread(target=control, args=(_data, lock, stop_event))
            thread2.start()
            startPlot(_data, lock, stop_event)
            stop_event.set()
            thread2.join()

            print("Plot closed. Exiting program.")
        else:
            thread1 = threading.Thread(target=control, args=(_data, lock, stop_event))
            thread1.start()
            thread1.join()

    except KeyboardInterrupt:
        print("Interrupted. Stopping motor.")
    except Exception as error:
        print("UNKNOWN ERROR OCCURRED")
        print(error)
    finally:
        stop_event.set()
        stop_motor()
