import zmq
import Jetson.GPIO as GPIO
import time

# GPIOピン設定
SERVO_PIN = 32  # BOARDモードの32ピン
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz
servo.start(7.5)  # 初期位置 (90度)

# ZMQソケットの設定
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:5555")

def set_servo_angle(angle):
    duty_cycle = 2.5 + (angle / 18.0)  # サーボモーターの角度を計算
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)

try:
    while True:
        message = socket.recv_string()
        direction = int(message)

        # 方向に応じてサーボモーターの角度を設定
        if direction == 0:
            set_servo_angle(45)  # 左
        elif direction == 1:
            set_servo_angle(90)  # 直進
        elif direction == 2:
            set_servo_angle(135)  # 右

        socket.send_string("ACK")  # 応答
except KeyboardInterrupt:
    print("終了します...")
finally:
    servo.stop()
    GPIO.cleanup()

