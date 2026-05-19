#include "QUBE.hpp"

QUBE qube;

const unsigned long COMMAND_TIMEOUT_MS = 100;
unsigned long lastCommandTime = 0;

void setup()
{
  Serial.begin(115200);
  qube.begin();
  qube.resetMotorEncoder();
  qube.resetPendulumEncoder();
  qube.setMotorVoltage(0);
  qube.setRGB(0, 999, 999);
  qube.update();
}

bool receiveData()
{
  if (Serial.available() >= 10)
  {
    bool resetMotorEncoder = Serial.read();
    bool resetPendulumEncoder = Serial.read();

    int r_MSB = Serial.read();
    int r_LSB = Serial.read();
    int r = (r_MSB << 8) + r_LSB;

    int g_MSB = Serial.read();
    int g_LSB = Serial.read();
    int g = (g_MSB << 8) + g_LSB;

    int b_MSB = Serial.read();
    int b_LSB = Serial.read();
    int b = (b_MSB << 8) + b_LSB;

    int motorCommand_MSB = Serial.read();
    int motorCommand_LSB = Serial.read();
    int motorCommand = ((motorCommand_MSB << 8) + motorCommand_LSB) - 999;

    if (resetMotorEncoder)
    {
      qube.resetMotorEncoder();
    }
    if (resetPendulumEncoder)
    {
      qube.resetPendulumEncoder();
    }

    qube.setRGB(r, g, b);
    qube.setMotorSpeed(motorCommand);
    lastCommandTime = millis();
    return true;
  }

  if (millis() - lastCommandTime > COMMAND_TIMEOUT_MS)
  {
    qube.setMotorSpeed(0);
  }

  return false;
}

void sendEncoderData(bool encoder)
{
  float encoderAngle = 0;

  if (encoder == 1)
  {
    encoderAngle = qube.getPendulumAngle(false);
  }
  else
  {
    encoderAngle = qube.getMotorAngle(false);
  }

  long revolutions = (long)encoderAngle / 360.0;

  float _angle = encoderAngle - revolutions * 360.0;
  long angle = (long)_angle;
  long angleDecimal = (_angle - angle) * 100;

  if (encoderAngle < 0)
  {
    revolutions = abs(revolutions);
    angle = abs(angle);
    angleDecimal = abs(angleDecimal);
    revolutions |= (1 << 15);
  }

  angle = (angle << 7) | angleDecimal;

  Serial.write(highByte(revolutions));
  Serial.write(lowByte(revolutions));
  Serial.write(highByte(angle));
  Serial.write(lowByte(angle));
}

void sendRPMData()
{
  long rpm = (long)qube.getRPM();
  bool dir = rpm < 0;

  if (dir)
  {
    rpm = abs(rpm);
    rpm |= 1 << 15;
  }

  Serial.write(highByte(rpm));
  Serial.write(lowByte(rpm));
}

void sendCurrentData()
{
  long current = abs((long)qube.getMotorCurrent());

  Serial.write(highByte(current));
  Serial.write(lowByte(current));
}

void sendData()
{
  sendEncoderData(0);
  sendEncoderData(1);
  sendRPMData();
  sendCurrentData();
}

void loop()
{
  qube.update();

  bool received = false;
  while (!received)
  {
    received = receiveData();
    qube.update();
  }

  sendData();
}
