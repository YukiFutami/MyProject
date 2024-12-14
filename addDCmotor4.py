import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import Jetson.GPIO as GPIO
import keyboard  # キーボードの入力を処理するためのライブラリ

# モデルのロード
white_line_model = load_model("my_model3.h5")
direction_model = load_model("trained_model.h5")

# GPIOの初期化
GPIO.setmode(GPIO.BOARD)
SERVO_PIN = 32
DC_MOTOR_PWM_PIN = 33
DC_MOTOR_DIR_PIN1_A = 29
DC_MOTOR_DIR_PIN2_A = 31
DC_MOTOR_DIR_PIN1_B = 35
DC_MOTOR_DIR_PIN2_B = 37

# DCモーターのGPIO設定
GPIO.setup(DC_MOTOR_DIR_PIN1_A, GPIO.OUT)
GPIO.setup(DC_MOTOR_DIR_PIN2_A, GPIO.OUT)
GPIO.setup(DC_MOTOR_DIR_PIN1_B, GPIO.OUT)
GPIO.setup(DC_MOTOR_DIR_PIN2_B, GPIO.OUT)
GPIO.setup(DC_MOTOR_PWM_PIN, GPIO.OUT)

# サーボモータのPWM設定
servo_pwm = GPIO.PWM(SERVO_PIN, 50)  # 50HzのPWM信号を設定
servo_pwm.start(7.5)  # 中央位置の信号

# DCモータのPWM設定
dc_motor_pwm = GPIO.PWM(DC_MOTOR_PWM_PIN, 100)  # 100HzのPWM信号を設定
dc_motor_pwm.start(0)  # 初期速度0

# キーボードの入力フラグ
is_driving_forward = False

# キーボードの入力処理
def check_keyboard_input():
    global is_driving_forward
    if keyboard.is_pressed('s'):
        is_driving_forward = True
    elif keyboard.is_pressed('x'):
        is_driving_forward = False

# testPic フォルダ内のすべての画像を処理して方向を決定する
def process_images_in_folder(folder_path):
    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        image = cv2.imread(image_path)

        # 前処理
        resized_image = cv2.resize(image, (256, 256))
        input_image = np.expand_dims(resized_image, axis=0) / 255.0

        # 白線認識モデルによる予測
        white_line_prediction = white_line_model.predict(input_image)

        # 方向認識モデルによる予測
        direction_prediction = direction_model.predict(input_image)
        direction_class = np.argmax(direction_prediction, axis=1)[0]  # クラスを取得

        # 予測結果に基づく車両制御
        control_vehicle(white_line_prediction, direction_class, image_filename)

# 車両制御
def control_vehicle(white_line_prediction, direction_class, image_filename):
    global is_driving_forward
    result_message = ""
    speed = 50  # デフォルト速度（50%デューティ比）

    if direction_class == 0:  # "straight"
        set_servo_angle(90)  # 中央
        if is_driving_forward:
            set_dc_motor_speed(1, 1, speed)  # 両モーター前進
        else:
            set_dc_motor_speed(0, 0, 0)  # 停止
        result_message = f"Image: {image_filename}, Direction: Straight"
    elif direction_class == 1:  # "left"
        set_servo_angle(120)  # 左方向
        if is_driving_forward:
            set_dc_motor_speed(1, 1, speed)  # 両モーター前進
        else:
            set_dc_motor_speed(0, 0, 0)  # 停止
        result_message = f"Image: {image_filename}, Direction: Left"
    elif direction_class == 2:  # "right"
        set_servo_angle(60)  # 右方向
        if is_driving_forward:
            set_dc_motor_speed(1, 1, speed)  # 両モーター前進
        else:
            set_dc_motor_speed(0, 0, 0)  # 停止
        result_message = f"Image: {image_filename}, Direction: Right"

    print(result_message)

def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18.0)
    servo_pwm.ChangeDutyCycle(duty_cycle)

def set_dc_motor_speed(direction_a, direction_b, speed_percentage):
    """
    DCモーターAとBの制御。
    direction_a, direction_b: 1=前進, 0=停止, -1=後退
    speed_percentage: 0~100のデューティ比
    """
    # モーターAの制御
    if direction_a == 0:
        GPIO.output(DC_MOTOR_DIR_PIN1_A, GPIO.LOW)
        GPIO.output(DC_MOTOR_DIR_PIN2_A, GPIO.LOW)
    elif direction_a == 1:
        GPIO.output(DC_MOTOR_DIR_PIN1_A, GPIO.HIGH)
        GPIO.output(DC_MOTOR_DIR_PIN2_A, GPIO.LOW)
    elif direction_a == -1:
        GPIO.output(DC_MOTOR_DIR_PIN1_A, GPIO.LOW)
        GPIO.output(DC_MOTOR_DIR_PIN2_A, GPIO.HIGH)

    # モーターBの制御
    if direction_b == 0:
        GPIO.output(DC_MOTOR_DIR_PIN1_B, GPIO.LOW)
        GPIO.output(DC_MOTOR_DIR_PIN2_B, GPIO.LOW)
    elif direction_b == 1:
        GPIO.output(DC_MOTOR_DIR_PIN1_B, GPIO.HIGH)
        GPIO.output(DC_MOTOR_DIR_PIN2_B, GPIO.LOW)
    elif direction_b == -1:
        GPIO.output(DC_MOTOR_DIR_PIN1_B, GPIO.LOW)
        GPIO.output(DC_MOTOR_DIR_PIN2_B, GPIO.HIGH)

    # PWMによる速度制御
    dc_motor_pwm.ChangeDutyCycle(speed_percentage)

# メイン関数
def main():
    try:
        folder_path = "testPic"  # 画像フォルダのパス
        while True:
            check_keyboard_input()
            process_images_in_folder(folder_path)
    except KeyboardInterrupt:
        print("\n終了します...")
    finally:
        servo_pwm.stop()
        dc_motor_pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()

