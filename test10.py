import Jetson.GPIO as GPIO
import time
import subprocess
import cv2
import numpy as np
import threading
import os
from pynput import keyboard

# 保存先フォルダ作成（存在しない場合）
output_dir = "Pictures6"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder created: {output_dir}")

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
servo.start(7.5)  # 初期位置90°に設定

# DCモーターのセットアップ
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1_A, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_A, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1_B, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_B, GPIO.OUT)

dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 周波数1kHzでPWM制御
dc_motor_pwm.start(0)  # PWMを0%で開始

# サーボモーターを指定した角度に動かす関数
def set_servo_angle(angle):
    duty_cycle = max(2, min(12, 2 + (angle / 18)))  # 角度に応じてデューティサイクルを設定
    print(f"Setting servo to {angle}° with duty cycle: {duty_cycle}%")
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # モーターが動く時間を確保
    servo.ChangeDutyCycle(0)  # PWM信号を停止

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

# 両方のDCモーターを制御する関数
def set_both_dc_motors(speed=0, direction="stop"):
    set_dc_motor(speed, direction, "A")
    set_dc_motor(speed, direction, "B")

# キー入力に応じてサーボとDCモーターを制御する関数
def on_press(key):
    try:
        if key.char == 'j':  # 'j'キーでサーボを135°に設定
            set_servo_angle(125)
        elif key.char == 'k':  # 'k'キーでサーボを50°に設定
            set_servo_angle(60)
        elif key.char == 'r':  # 'r'キーでサーボを90°に戻す
            set_servo_angle(90)
        elif key.char == 'w':  # 'w'キーで前進
            set_both_dc_motors(30, "forward")
        elif key.char == 's':  # 's'キーで後退
            set_both_dc_motors(30, "backward")
        elif key.char == 'a':  # 'a'キーで左回転
            set_dc_motor(60, "forward", "A")
            set_dc_motor(60, "backward", "B")
        elif key.char == 'd':  # 'd'キーで右回転
            set_dc_motor(60, "forward", "B")
            set_dc_motor(60, "backward", "A")
        elif key.char == 'x':  # 'x'キーで停止
            set_both_dc_motors(0, "stop")
    except AttributeError:
        pass

# カメラから画像を保存するバックグラウンドスレッド
def save_images_from_camera():
    cap = cv2.VideoCapture(0)  # カメラを開く

    if not cap.isOpened():
        print("Error: Camera not detected. Exiting camera thread.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_filename = os.path.join(output_dir, f"captured_image_{frame_count}.png")
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
        time.sleep(0.1)  # 負荷軽減のため100ms待機
except KeyboardInterrupt:
    print("\nProgram interrupted.")
finally:
    listener.stop()  # キーリスナーを停止
    image_thread.join()  # スレッドを終了
    servo.stop()  # サーボのPWMを停止
    dc_motor_pwm.stop()  # DCモーターのPWMを停止
    GPIO.cleanup()  # GPIOをクリーンアップ

