import Jetson.GPIO as GPIO
import time
import cv2
import numpy as np
from keras.models import load_model
import os
import tensorflow as tf

# メモリ制限を設定
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Set memory growth for GPU.")
    except RuntimeError as e:
        print(e)

# GPIOの設定
GPIO.cleanup()
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
def set_dc_motor(speed=0, direction="forward", motor="A"):
    if motor == "A" or motor == "B":
        if direction == "forward":
            GPIO.output(dc_motor_dir_pin1_A if motor == "A" else dc_motor_dir_pin1_B, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2_A if motor == "A" else dc_motor_dir_pin2_B, GPIO.LOW)
        elif direction == "stop":
            GPIO.output(dc_motor_dir_pin1_A if motor == "A" else dc_motor_dir_pin1_B, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2_A if motor == "A" else dc_motor_dir_pin2_B, GPIO.LOW)
    dc_motor_pwm.ChangeDutyCycle(speed)  # PWMで速度を設定

# 両方のDCモーターを制御する関数
def set_both_dc_motors(speed=0, direction="forward"):
    set_dc_motor(speed, direction, "A")
    set_dc_motor(speed, direction, "B")

# 白線認識モデルと方向指示モデルのロード
line_model = load_model('my_model3.h5')
direction_model = load_model('trained_model.h5')

# 前回の方向予測を保持する変数
previous_direction = None

# 指定のフォルダから画像を読み込み認識処理を行う関数
def recognize_lines_from_images(folder_path):
    # モデルの準備が完了したら「スタート」メッセージを表示
    print("Model is ready. Press 'w' to start moving.")

    for img_file in os.listdir(folder_path):
        if img_file.endswith(('.jpg', '.png')):
            img_path = os.path.join(folder_path, img_file)
            print(f"Processing {img_path}")
            image = cv2.imread(img_path)

            # 白線の認識
            line_image = recognize_lines(image, line_model)

            # 方向指示の予測
            try:
                direction = predict_direction(line_image, direction_model)
            except:
                # 方向予測が失敗した場合、前回の方向を使用
                print("Direction prediction failed. Using previous direction.")
                direction = previous_direction

            # 認識結果をサーボに反映
            set_servo_angle(direction)

            # 認識が完了したらDCモーターを前進させる
            set_both_dc_motors(30, "forward")  # PWM 30%で前進

            # 前回の方向を更新
            previous_direction = direction

    # 終了処理
    print("All images processed. Press 'x' to stop.")
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x'):
        set_both_dc_motors(0, "stop")  # 停止
        GPIO.cleanup()

def recognize_lines(image, model):
    input_image = cv2.resize(image, (256, 256)) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    prediction = model.predict(input_image)
    prediction = np.squeeze(prediction)

    lines = cv2.HoughLinesP(
        prediction.astype(np.uint8), 1, np.pi/180, threshold=50,
        minLineLength=50, maxLineGap=10
    )

    if lines is None or len(lines) == 0:
        print("No lines detected.")
        return image

    for line in lines:
        line = line[0]
        cv2.line(image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)

    return image

def predict_direction(image, model):
    # 方向を予測するための前処理
    resized_image = cv2.resize(image, (256, 256))
    normalized_image = resized_image / 255.0
    input_data = np.expand_dims(normalized_image, axis=0)

    # モデルに通して方向を予測
    prediction = model.predict(input_data)
    direction_label = np.argmax(prediction)  # 最も高い確率を持つクラス
    if direction_label == 0:  # クラス "straight"
        set_servo_angle(85)
        result_message = f"Direction: Straight"
    elif direction_label == 1:  # クラス "left"
        set_servo_angle(110)
        result_message = f"Direction: Left"
    elif direction_label == 2:  # クラス "right"
        set_servo_angle(70)
        result_message = f"Direction: Right"
    else:
        result_message = "No valid direction predicted"
    
    print(result_message)
    return direction_label

# 指定のフォルダ内の画像を使用して認識と方向予測を実行
recognize_lines_from_images('testPic')

