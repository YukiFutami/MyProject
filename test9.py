# DCモーターで回転し向きを変える
# サーボモーターが反応しない

import Jetson.GPIO as GPIO
import time
import subprocess
import cv2
import numpy as np
import threading
from pynput import keyboard

# GPIOの設定
GPIO.cleanup()  # GPIOをリセット
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# サーボモーターのピン設定
servo_pin = 32  # サーボモーターのPWMピン

# DCモーターのピン設定
dc_motor_pwm_pin = 33         # PWM信号で両方のモーターの速度を制御
dc_motor_dir_pin1_A = 29      # モーターAの方向制御ピン1
dc_motor_dir_pin2_A = 31      # モーターAの方向制御ピン2
dc_motor_dir_pin1_B = 35      # モーターBの方向制御ピン1
dc_motor_dir_pin2_B = 37      # モーターBの方向制御ピン2

# サーボモーターのセットアップ
GPIO.setup(servo_pin, GPIO.OUT)
servo = GPIO.PWM(servo_pin, 50)  # 50HzでPWM信号を送る（サーボの一般的な周波数）

# DCモーターのセットアップ
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1_A, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_A, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1_B, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_B, GPIO.OUT)

dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # DCモーターのPWM信号、周波数1kHz
dc_motor_pwm.start(0)  # PWMを0%で開始

servo.start(0)  # サーボモーターのPWMを0%で開始

# サーボモーターを指定した角度に動かす関数
def set_servo_angle(angle):
    duty_cycle = max(2, min(12, 2 + (angle / 18)))  # 角度に応じてデューティサイクルを設定
    print(f"Setting servo to {angle}° with duty cycle: {duty_cycle}%")
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # モーターが動く時間を確保
    servo.ChangeDutyCycle(0)  # PWM信号を停止してリセット

# DCモーターの動作制御関数
def set_dc_motor(speed=0, direction="stop", motor="A"):
    if motor == "A":
        if direction == "forward":
            GPIO.output(dc_motor_dir_pin1_A, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2_A, GPIO.LOW)
        elif direction == "backward":
            GPIO.output(dc_motor_dir_pin1_A, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2_A, GPIO.HIGH)
        elif direction == "stop":
            GPIO.output(dc_motor_dir_pin1_A, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2_A, GPIO.LOW)
    elif motor == "B":
        if direction == "forward":
            GPIO.output(dc_motor_dir_pin1_B, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2_B, GPIO.LOW)
        elif direction == "backward":
            GPIO.output(dc_motor_dir_pin1_B, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2_B, GPIO.HIGH)
        elif direction == "stop":
            GPIO.output(dc_motor_dir_pin1_B, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2_B, GPIO.LOW)

    dc_motor_pwm.ChangeDutyCycle(speed)  # PWMで速度を設定

# キー入力に応じてサーボとDCモーターを制御する関数
def on_press(key):
    try:
        if key.char == 'j':  # 'j'キーでサーボを135°に設定
            print("Key 'j' pressed: Moving servo to 135°")
            set_servo_angle(135)
        elif key.char == 'k':  # 'k'キーでサーボを50°に設定
            print("Key 'k' pressed: Moving servo to 50°")
            set_servo_angle(50)
        elif key.char == 'r':  # 'r'キーでサーボを90°に戻す
            print("Key 'r' pressed: Moving servo to 90°")
            set_servo_angle(90)
        elif key.char == 'w':  # 'w'キーでモーターA、Bを前進
            set_dc_motor(10, "forward", "A")
            set_dc_motor(10, "forward", "B")
        elif key.char == 's':  # 's'キーでモーターA、Bを後退
            set_dc_motor(10, "backward", "A")
            set_dc_motor(10, "backward", "B")
        elif key.char == 'a':  # 'a'キーでモーターAのみ前進
            set_dc_motor(10, "forward", "A")
            set_dc_motor(10, "backward", "B")
        elif key.char == 'd':  # 'd'キーでモーターBのみ前進
            set_dc_motor(10, "forward", "B")
            set_dc_motor(10, "backward", "A")
        elif key.char == 'x':  # 'x'キーでモーターA、Bを停止
            print("Key 'x' pressed: Stopping both motors")
            set_dc_motor(0, "stop", "A")
            set_dc_motor(0, "stop", "B")
    except AttributeError:
        pass


# カメラから画像を保存するバックグラウンドスレッド
def save_images_from_camera():
    cap = cv2.VideoCapture(0)  # カメラを開く

    if not cap.isOpened():
        print("Error: Camera not detected")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_filename = f"captured_image_{frame_count}.png"
        cv2.imwrite(image_filename, frame)  # 画像を保存
        print(f"Image saved: {image_filename}")
        frame_count += 1
        time.sleep(1)  # 1秒ごとに保存

    cap.release()

# キーリスナーを開始
listener = keyboard.Listener(on_press=on_press)
listener.start()

# 画像保存スレッドを開始
image_thread = threading.Thread(target=save_images_from_camera)
image_thread.daemon = True  # プログラム終了時にスレッドも終了
image_thread.start()

# プログラムが終了しないように待機
try:
    while True:
        pass
except KeyboardInterrupt:
    print("\nProgram interrupted.")
finally:
    listener.stop()  # キーリスナーを停止
    servo.stop()  # サーボのPWMを停止
    dc_motor_pwm.stop()  # DCモーターのPWMを停止
    GPIO.cleanup()  # GPIOをクリーンアップ