import Jetson.GPIO as GPIO
import time

SERVO_PIN = 32
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(7.5)  # 中央位置

try:
    while True:
        pwm.ChangeDutyCycle(12.5)  # 右
        time.sleep(1)
        pwm.ChangeDutyCycle(2.5)  # 左
        time.sleep(1)
        pwm.ChangeDutyCycle(7.5)  # 中央
        time.sleep(1)
except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()

