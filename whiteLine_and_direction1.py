import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import Jetson.GPIO as GPIO

# モデルのロード
white_line_model = load_model("my_model3.h5")
direction_model = load_model("trained_model.h5")

# GPIOの初期化
GPIO.setmode(GPIO.BOARD)
SERVO_PIN = 32
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50HzのPWM信号を設定
pwm.start(7.5)  # 中央位置の信号

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
    # direction_class に基づく制御
    result_message = ""
    if direction_class == 0:  # "straight"
        set_servo_angle(90)  # 中央
        result_message = f"Image: {image_filename}, Direction: Straight"
    elif direction_class == 1:  # "left"
        set_servo_angle(120)  # 左方向
        result_message = f"Image: {image_filename}, Direction: Left"
    elif direction_class == 2:  # "right"
        set_servo_angle(60)  # 右方向
        result_message = f"Image: {image_filename}, Direction: Right"
    
    print(result_message)

def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18.0)
    pwm.ChangeDutyCycle(duty_cycle)

if __name__ == "__main__":
    try:
        # testPic フォルダのパス
        folder_path = "testPic"

        # フォルダ内のすべての画像を処理
        process_images_in_folder(folder_path)
    finally:
        # 終了時にPWMを停止する
        pwm.stop()
        GPIO.cleanup()

