import os
import Jetson.GPIO as GPIO
import tensorflow as tf
import time
from keras.models import load_model
from PIL import Image
import numpy as np

# GPIOピンの設定
SERVO_PIN = 32
DC_MOTOR_PWM_PIN = 33
DC_MOTOR_A_PIN = 29
DC_MOTOR_B_PIN = 31

# TensorFlowのメモリ制限設定
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]  # メモリ制限 (MB)
            )
    except RuntimeError as e:
        print(e)

# モデルの読み込み
MODEL_PATH = "my_model.h5"
model = load_model(MODEL_PATH)

# GPIO設定
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(DC_MOTOR_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_MOTOR_A_PIN, GPIO.OUT)
GPIO.setup(DC_MOTOR_B_PIN, GPIO.OUT)

# PWM初期化
servo_pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz (20ms周期)
dc_motor_pwm = GPIO.PWM(DC_MOTOR_PWM_PIN, 100)  # 100Hz
servo_pwm.start(7.5)  # サーボを90度に初期化
dc_motor_pwm.start(0)  # モーターを停止

def process_images_in_folder(folder_path):
    """フォルダ内の画像を処理し、モデルで予測する"""
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path).resize((256, 256))
            image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
            prediction = model.predict(image_array)

            if prediction[0][0] > 0.5:
                print(f"Image: {file_name}, Action: Turn Left")
                turn_left()
            elif prediction[0][1] > 0.5:
                print(f"Image: {file_name}, Action: Turn Right")
                turn_right()
            else:
                print(f"Image: {file_name}, Action: Move Forward")
                move_forward()

            time.sleep(0.5)  # 操作間の遅延

def move_forward():
    GPIO.output(DC_MOTOR_A_PIN, GPIO.HIGH)
    GPIO.output(DC_MOTOR_B_PIN, GPIO.LOW)
    dc_motor_pwm.ChangeDutyCycle(30)
    servo_pwm.ChangeDutyCycle(7.5)  # サーボを中央に設定

def turn_left():
    GPIO.output(DC_MOTOR_A_PIN, GPIO.HIGH)
    GPIO.output(DC_MOTOR_B_PIN, GPIO.LOW)
    dc_motor_pwm.ChangeDutyCycle(30)
    servo_pwm.ChangeDutyCycle(10)  # サーボを左に回転

def turn_right():
    GPIO.output(DC_MOTOR_A_PIN, GPIO.HIGH)
    GPIO.output(DC_MOTOR_B_PIN, GPIO.LOW)
    dc_motor_pwm.ChangeDutyCycle(30)
    servo_pwm.ChangeDutyCycle(5)  # サーボを右に回転

def stop():
    GPIO.output(DC_MOTOR_A_PIN, GPIO.LOW)
    GPIO.output(DC_MOTOR_B_PIN, GPIO.LOW)
    dc_motor_pwm.ChangeDutyCycle(0)

def check_keyboard_input():
    """キーボード入力をチェックする"""
    global is_driving_forward
    key = input("Press 's' to start driving, 'x' to stop: ").strip()
    if key == 's':
        is_driving_forward = True
    elif key == 'x':
        is_driving_forward = False

def main():
    try:
        global is_driving_forward
        is_driving_forward = False

        while True:
            check_keyboard_input()
            if is_driving_forward:
                process_images_in_folder("testPic")
            else:
                stop()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # PWM停止とGPIOクリーンアップ
        servo_pwm.stop()
        dc_motor_pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()

