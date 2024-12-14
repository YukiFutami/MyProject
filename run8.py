import os
import time
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import Jetson.GPIO as GPIO

# モデルの読み込み
model = load_model("my_model.h5")

# GPIOピン設定
SERVO_PIN = 32
DC_PWM_PIN = 33
DC_A_PIN1 = 29
DC_A_PIN2 = 31
DC_B_PIN1 = 35
DC_B_PIN2 = 37

# GPIOの初期設定
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(DC_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_A_PIN1, GPIO.OUT)
GPIO.setup(DC_A_PIN2, GPIO.OUT)
GPIO.setup(DC_B_PIN1, GPIO.OUT)
GPIO.setup(DC_B_PIN2, GPIO.OUT)

# サーボモーターとDCモーターのPWM制御
servo_pwm = GPIO.PWM(SERVO_PIN, 50)  # サーボは50Hz
dc_motor_pwm = GPIO.PWM(DC_PWM_PIN, 100)  # DCモーターは100Hz

# サーボモーターを90度（直進）に設定
servo_pwm.start(7.5)  # デューティサイクル7.5が90度
print("サーボモーターを90度（直進位置）に設定しました。")

# DCモーター初期設定
dc_motor_pwm.start(50)  # DCモーター初期速度（50%デューティ）
GPIO.output(DC_A_PIN1, GPIO.HIGH)
GPIO.output(DC_A_PIN2, GPIO.LOW)
GPIO.output(DC_B_PIN1, GPIO.HIGH)
GPIO.output(DC_B_PIN2, GPIO.LOW)

def predict_direction(image_path):
    """画像を処理して進行方向を決定"""
    img = Image.open(image_path).resize((256, 256)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]  # 配列を1次元に
    if prediction[0] > 0.5:
        return "straight"
    elif prediction[1] > 0.5:
        return "left"
    elif prediction[2] > 0.5:
        return "right"
    else:
        return "unknown"

def set_servo_angle(direction):
    """進行方向に応じてサーボモーターの角度を変更"""
    if direction == "straight":
        servo_pwm.ChangeDutyCycle(7.5)  # 直進
        print("Direction: Straight")
    elif direction == "left":
        servo_pwm.ChangeDutyCycle(10)  # 左折
        print("Direction: Left")
    elif direction == "right":
        servo_pwm.ChangeDutyCycle(5)  # 右折
        print("Direction: Right")
    else:
        print("Direction: Unknown")
        servo_pwm.ChangeDutyCycle(7.5)  # デフォルト直進

try:
    # 画像フォルダの設定
    image_folder = "Picture"
    image_files = [f for f in sorted(os.listdir(image_folder)) if os.path.isfile(os.path.join(image_folder, f))]

    print(f"Found {len(image_files)} images. Processing in order...")
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        # 進行方向を認識
        direction = predict_direction(image_path)

        # サーボモーターの角度を設定
        set_servo_angle(direction)

        # 画像処理間隔
        time.sleep(1)

except KeyboardInterrupt:
    print("Program interrupted.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # GPIOを解放
    try:
        if 'servo_pwm' in locals():
            servo_pwm.stop()
        if 'dc_motor_pwm' in locals():
            dc_motor_pwm.stop()
    except Exception as e:
        print(f"Failed to stop PWM: {e}")
    finally:
        GPIO.cleanup()

