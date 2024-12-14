import os
import numpy as np
import tensorflow as tf
import Jetson.GPIO as GPIO
from time import sleep

# GPIOピン設定
SERVO_PIN = 32
DC_PWM_PIN = 33
DC_A_PIN1 = 29
DC_A_PIN2 = 31
DC_B_PIN1 = 35
DC_B_PIN2 = 37

# GPIOの初期化
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(DC_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_A_PIN1, GPIO.OUT)
GPIO.setup(DC_A_PIN2, GPIO.OUT)
GPIO.setup(DC_B_PIN1, GPIO.OUT)
GPIO.setup(DC_B_PIN2, GPIO.OUT)

servo = GPIO.PWM(SERVO_PIN, 50)  # サーボモーター PWM 50Hz
dc_motor = GPIO.PWM(DC_PWM_PIN, 100)  # DCモーター PWM 100Hz
servo.start(7.5)  # サーボモーターを中央位置に初期化
dc_motor.start(30)  # DCモーターの初期速度（30%デューティサイクル）

# モデルのロード
model = tf.keras.models.load_model("my_model.h5")

# 進行方向の設定関数
def set_motor_direction(direction):
    if direction == "forward":
        GPIO.output(DC_A_PIN1, GPIO.HIGH)
        GPIO.output(DC_A_PIN2, GPIO.LOW)
        GPIO.output(DC_B_PIN1, GPIO.HIGH)
        GPIO.output(DC_B_PIN2, GPIO.LOW)
        servo.ChangeDutyCycle(7.5)  # 中央
    elif direction == "left":
        GPIO.output(DC_A_PIN1, GPIO.HIGH)
        GPIO.output(DC_A_PIN2, GPIO.LOW)
        GPIO.output(DC_B_PIN1, GPIO.HIGH)
        GPIO.output(DC_B_PIN2, GPIO.LOW)
        servo.ChangeDutyCycle(10.0)  # 左折
    elif direction == "right":
        GPIO.output(DC_A_PIN1, GPIO.HIGH)
        GPIO.output(DC_A_PIN2, GPIO.LOW)
        GPIO.output(DC_B_PIN1, GPIO.HIGH)
        GPIO.output(DC_B_PIN2, GPIO.LOW)
        servo.ChangeDutyCycle(5.0)  # 右折

# 推論と進行方向の判定
def determine_direction(image_path):
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        if predicted_class == 0:  # クラス0: 左折
            return "left"
        elif predicted_class == 1:  # クラス1: 直進
            return "forward"
        elif predicted_class == 2:  # クラス2: 右折
            return "right"
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# メイン処理
def main():
    image_folder = "Picture"
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in folder.")
        return
    
    print(f"Found {len(image_files)} images. Processing...")
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing {image_path}...")
        
        direction = determine_direction(image_path)
        if direction:
            print(f"Direction: {direction}")
            set_motor_direction(direction)
        sleep(1)  # 次の画像に進む前に1秒待機

    print("Processing completed.")

# 実行
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        servo.stop()
        dc_motor.stop()
        GPIO.cleanup()
        print("GPIO cleanup completed.")

