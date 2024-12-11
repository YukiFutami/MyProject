import os
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import Jetson.GPIO as GPIO

# GPIO設定
SERVO_PIN = 32  # サーボモーター用
DC_MOTOR_PWM_PIN = 33  # DCモーターの速度制御用
DC_MOTOR_DIR_PIN1_A = 29  # DCモーターAの方向制御用
DC_MOTOR_DIR_PIN2_A = 31  # DCモーターAの方向制御用
DC_MOTOR_DIR_PIN1_B = 35  # DCモーターBの方向制御用
DC_MOTOR_DIR_PIN2_B = 37  # DCモーターBの方向制御用

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(DC_MOTOR_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_MOTOR_DIR_PIN1_A, GPIO.OUT)
GPIO.setup(DC_MOTOR_DIR_PIN2_A, GPIO.OUT)
GPIO.setup(DC_MOTOR_DIR_PIN1_B, GPIO.OUT)
GPIO.setup(DC_MOTOR_DIR_PIN2_B, GPIO.OUT)

# サーボモーター設定
servo_pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz
servo_pwm.start(7.5)  # 初期位置90°

# DCモーター設定
dc_motor_pwm = GPIO.PWM(DC_MOTOR_PWM_PIN, 1000)  # 1kHz
dc_motor_pwm.start(0)

# モデル読み込み
model = load_model('my_model.h5')

# テスト用画像フォルダ
image_dir = 'Picture'

# スピードと角度の設定
STRAIGHT_SPEED = 70  # 直線の速度
CURVE_SPEED = 40  # 曲線の速度
ANGLE_CENTER = 7.5  # 直進時のPWMデューティ比
ANGLE_LEFT = 9.0  # 左折時のPWMデューティ比
ANGLE_RIGHT = 6.0  # 右折時のPWMデューティ比

# 入力画像の前処理
def preprocess_image(frame):
    img = cv2.resize(frame, (256, 256)) / 255.0
    return np.expand_dims(img, axis=0)

# ステアリング角度計算
def calculate_steering_angle(prediction):
    left_white = np.mean(prediction[:, :128])
    right_white = np.mean(prediction[:, 128:])
    
    if left_white > right_white:
        return 'LEFT'
    elif right_white > left_white:
        return 'RIGHT'
    return 'STRAIGHT'

# モーター制御
def control_motors(direction):
    if direction == 'LEFT':
        GPIO.output(DC_MOTOR_DIR_PIN1_A, GPIO.HIGH)
        GPIO.output(DC_MOTOR_DIR_PIN2_A, GPIO.LOW)
        GPIO.output(DC_MOTOR_DIR_PIN1_B, GPIO.LOW)
        GPIO.output(DC_MOTOR_DIR_PIN2_B, GPIO.HIGH)
        servo_pwm.ChangeDutyCycle(ANGLE_LEFT)
        dc_motor_pwm.ChangeDutyCycle(CURVE_SPEED)
    elif direction == 'RIGHT':
        GPIO.output(DC_MOTOR_DIR_PIN1_A, GPIO.LOW)
        GPIO.output(DC_MOTOR_DIR_PIN2_A, GPIO.HIGH)
        GPIO.output(DC_MOTOR_DIR_PIN1_B, GPIO.HIGH)
        GPIO.output(DC_MOTOR_DIR_PIN2_B, GPIO.LOW)
        servo_pwm.ChangeDutyCycle(ANGLE_RIGHT)
        dc_motor_pwm.ChangeDutyCycle(CURVE_SPEED)
    else:  # STRAIGHT
        GPIO.output(DC_MOTOR_DIR_PIN1_A, GPIO.HIGH)
        GPIO.output(DC_MOTOR_DIR_PIN2_A, GPIO.LOW)
        GPIO.output(DC_MOTOR_DIR_PIN1_B, GPIO.HIGH)
        GPIO.output(DC_MOTOR_DIR_PIN2_B, GPIO.LOW)
        servo_pwm.ChangeDutyCycle(ANGLE_CENTER)
        dc_motor_pwm.ChangeDutyCycle(STRAIGHT_SPEED)

try:
    print("キーボードで 's' を押してスタート、'q' を押して停止します。")

    while True:
        # キー入力待ち
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 開始
            print("スタートしました！1秒後に処理を開始します。")
            time.sleep(1)  # 1秒待機

            for img_name in sorted(os.listdir(image_dir)):
                img_path = os.path.join(image_dir, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"画像 {img_name} が読み込めませんでした。")
                    continue

                # 入力画像の前処理とモデル推論
                input_img = preprocess_image(img)
                prediction = model.predict(input_img)[0, :, :, 0]

                # ステアリング計算
                direction = calculate_steering_angle(prediction)

                # モーター制御
                control_motors(direction)

                # 表示
                pred_img = (prediction > 0.5).astype(np.uint8) * 255
                pred_img = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
                combined_img = np.hstack((img, pred_img))
                cv2.putText(combined_img, f"Direction: {direction}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Test Image', combined_img)

                print(f"画像: {img_name} | 方向: {direction}")

                # 画像を表示して1秒待機
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
            break

        elif key == ord('q'):  # 終了
            print("終了します。")
            break

finally:
    servo_pwm.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()

