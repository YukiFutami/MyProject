import Jetson.GPIO as GPIO
import time

# GPIOの設定
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# DCモーターAのピン設定
dc_motor_dir_pin1_A = 29
dc_motor_dir_pin2_A = 31

# DCモーターBのピン設定
dc_motor_dir_pin1_B = 35
dc_motor_dir_pin2_B = 37

# 共通のPWMピン
dc_motor_pwm_pin = 33

# GPIOの初期化
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1_A, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_A, GPIO.OUT)

GPIO.setup(dc_motor_dir_pin1_B, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_B, GPIO.OUT)

# PWMの初期化
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)
dc_motor_pwm.start(0)

def set_dc_motor(speed=0, direction="stop", motor="A"):
    if motor == "A":
        dir1 = dc_motor_dir_pin1_A
        dir2 = dc_motor_dir_pin2_A
    elif motor == "B":
        dir1 = dc_motor_dir_pin1_B
        dir2 = dc_motor_dir_pin2_B
    else:
        print("Invalid motor selection")
        return

    if direction == "forward":
        GPIO.output(dir1, GPIO.HIGH)
        GPIO.output(dir2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dir1, GPIO.LOW)
        GPIO.output(dir2, GPIO.HIGH)
    elif direction == "stop":
        GPIO.output(dir1, GPIO.LOW)
        GPIO.output(dir2, GPIO.LOW)
    else:
        print("Invalid direction")
        return

    dc_motor_pwm.ChangeDutyCycle(speed)

def stop_all_motors():
    set_dc_motor(0, "stop", "A")
    set_dc_motor(0, "stop", "B")

try:
    while True:
        command = input("Enter command (e.g., 'A forward 50', 'B stop', 'exit'): ")
        if command == "exit":
            break

        parts = command.split()
        if len(parts) == 3:
            motor = parts[0].upper()
            direction = parts[1].lower()
            speed = int(parts[2])

            if motor in ["A", "B"] and direction in ["forward", "backward", "stop"] and 0 <= speed <= 100:
                set_dc_motor(speed, direction, motor)
            else:
                print("Invalid command format or values.")
        else:
            print("Invalid command format. Use: '<motor> <direction> <speed>'")

except KeyboardInterrupt:
    pass
finally:
    stop_all_motors()
    dc_motor_pwm.stop()
    GPIO.cleanup()

