import os
import time
import numpy as np
import Jetson.GPIO as GPIO  # Jetson.GPIOを使用するためにインポート
import cv2  # OpenCVを使用してカメラからの画像取得を行う
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image

# GPIOのピン設定
SERVO_PWM_PIN = 32  # サーボモーターのPWMピン
DC_PWM_PIN = 33    # DCモーターのPWMピン
DC_MOTOR_A_PIN_1 = 29
DC_MOTOR_A_PIN_2 = 31
DC_MOTOR_B_PIN_1 = 35
DC_MOTOR_B_PIN_2 = 37

# モデルのロード
model = load_model('my_model.h5')

# 入力フォルダと出力フォルダの作成
input_folder = 'CapturedImages'
output_folder = 'Outputs'
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# GPIOの設定
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_MOTOR_A_PIN_1, GPIO.OUT)
GPIO.setup(DC_MOTOR_A_PIN_2, GPIO.OUT)
GPIO.setup(DC_MOTOR_B_PIN_1, GPIO.OUT)
GPIO.setup(DC_MOTOR_B_PIN_2, GPIO.OUT)

pwm_servo = GPIO.PWM(SERVO_PWM_PIN, 50)  # 50HzでPWM信号を送る
pwm_servo.start(90)  # サーボモーターを初期位置（90度）にする

pwm_dc_motor = GPIO.PWM(DC_PWM_PIN, 50)
pwm_dc_motor.start(30)  # DCモーターを一定のスピードで動かす

# スタートフラグ
start = False
paused = False  # 一時停止フラグ

def capture_image():
    cap = cv2.VideoCapture(0)  # カメラを取得
    ret, frame = cap.read()
    if ret:
        timestamp = int(time.time())
        filename = f'{input_folder}/camera_image_{timestamp}.png'
        cv2.imwrite(filename, frame)  # フレームをファイルとして保存
    cap.release()
    return filename

def process_image(image_path):
    try:
        # 現在のファイルを表示
        print(f"Processing {image_path}...")

        # 画像の読み込みと前処理
        image = load_img(image_path, target_size=(256, 256))
        image_array = img_to_array(image) / 255.0  # 正規化
        image_array = np.expand_dims(image_array, axis=0)

        # モデルによる予測
        prediction = model.predict(image_array)

        # 結果を保存
        output_path = os.path.join(output_folder, f"output_{os.path.basename(image_path)}")
        prediction_image = (prediction[0] * 255).astype('uint8')  # スケールを戻す
        Image.fromarray(prediction_image.squeeze(), mode="L").save(output_path)

        print(f"Processed {image_path}, saved to {output_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def control_motors(direction):
    # 直進
    if direction == 'straight':
        GPIO.output(DC_MOTOR_A_PIN_1, GPIO.HIGH)
        GPIO.output(DC_MOTOR_A_PIN_2, GPIO.LOW)
        GPIO.output(DC_MOTOR_B_PIN_1, GPIO.HIGH)
        GPIO.output(DC_MOTOR_B_PIN_2, GPIO.LOW)
    # 左折
    elif direction == 'left':
        GPIO.output(DC_MOTOR_A_PIN_1, GPIO.HIGH)
        GPIO.output(DC_MOTOR_A_PIN_2, GPIO.LOW)
        GPIO.output(DC_MOTOR_B_PIN_1, GPIO.LOW)
        GPIO.output(DC_MOTOR_B_PIN_2, GPIO.HIGH)
    # 右折
    elif direction == 'right':
        GPIO.output(DC_MOTOR_A_PIN_1, GPIO.LOW)
        GPIO.output(DC_MOTOR_A_PIN_2, GPIO.HIGH)
        GPIO.output(DC_MOTOR_B_PIN_1, GPIO.HIGH)
        GPIO.output(DC_MOTOR_B_PIN_2, GPIO.LOW)

def main():
    global start, paused
    try:
        while True:
            if start and not paused:
                image_path = capture_image()
                process_image(image_path)
                # モデルからの出力に基づき、方向を制御
                direction = determine_direction(image_path)
                control_motors(direction)
                time.sleep(1)  # 1秒間隔で新しい画像を取得

            # キーボード入力による終了処理
            key = input("Press 'x' to exit, 's' to start or 'x' to pause/resume: ")
            if key == 'x':
                if start:
                    pwm_servo.ChangeDutyCycle(7.5)  # サーボモーターを90度に戻す
                    pwm_dc_motor.stop()  # モーターのPWMを停止
                    GPIO.cleanup()  # GPIOのリセット
                    print("Program terminated.")
                break
            elif key == 's':
                start = not start  # スタートフラグを切り替える
                paused = False  # スタート時に一時停止を解除する
                if start:
                    print("Program started.")
                else:
                    print("Program paused.")
            elif key == 'x' and not start:
                paused = not paused  # 一時停止状態を切り替える

    except KeyboardInterrupt:
        pwm_servo.ChangeDutyCycle(7.5)  # サーボモーターを90度に戻す
        pwm_dc_motor.stop()  # モーターのPWMを停止
        GPIO.cleanup()  # GPIOのリセット
        print("Program interrupted.")

def determine_direction(image_path):
    # モデルの出力を元に方向を決定する
    # 曲線の方向（左折、右折）を判定するためのロジック
    try:
        image = load_img(image_path, target_size=(256, 256))
        image_array = img_to_array(image) / 255.0  # 正規化
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)

        # 出力の処理
        if prediction[0][0] > 0.5:
            return 'right'  # 右折
        else:
            return 'left'  # 左折
    except Exception as e:
        print(f"Error determining direction: {e}")
        return 'straight'  # 直進がデフォルト

if __name__ == "__main__":
    main()

