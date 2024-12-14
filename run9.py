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

# 画像フォルダの設定
image_folder = "Picture"
images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

try:
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        img = Image.open(image_path).resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # モデルの予測
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        # 予測に基づく制御
        if predicted_class == 0:
            print(f"{image_name}: 直進")
            servo_pwm.ChangeDutyCycle(7.5)  # 直進
        elif predicted_class == 1:
            print(f"{image_name}: 左折")
            servo_pwm.ChangeDutyCycle(10.0)  # 左折
        elif predicted_class == 2:
            print(f"{image_name}: 右折")
            servo_pwm.ChangeDutyCycle(5.0)  # 右折

        time.sleep(0.5)  # 処理間隔

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # PWM停止とGPIO解放
    try:
        if servo_pwm:
            servo_pwm.stop()
        if dc_motor_pwm:
            dc_motor_pwm.stop()
    except Exception as pwm_error:
        print(f"PWM stop error: {pwm_error}")
    finally:
        try:
            GPIO.cleanup()
        except Exception as cleanup_error:
            print(f"GPIO cleanup error: {cleanup_error}")

