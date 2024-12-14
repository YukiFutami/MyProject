import RPi.GPIO as GPIO
import time
import threading
import cv2
from keras.models import load_model
import numpy as np

# GPIOピン設定
SERVO_PIN = 32
DC_PWM_PIN = 33
DC_A_PIN1 = 29
DC_A_PIN2 = 31
DC_B_PIN1 = 35
DC_B_PIN2 = 37
CAMERA_INDEX = 0
IMAGE_SAVE_PATH = "./images"
MODEL_PATH = "my_model.h5"  # 学習済みモデルのパス

# 初期化
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(DC_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_A_PIN1, GPIO.OUT)
GPIO.setup(DC_A_PIN2, GPIO.OUT)
GPIO.setup(DC_B_PIN1, GPIO.OUT)
GPIO.setup(DC_B_PIN2, GPIO.OUT)

# PWM設定
servo = GPIO.PWM(SERVO_PIN, 50)  # サーボモーター用 50Hz
servo.start(7.5)  # 初期位置 90度

dc_motor_pwm = GPIO.PWM(DC_PWM_PIN, 1000)  # DCモーター用 1000Hz
dc_motor_pwm.start(0)  # 初期状態は停止

# 学習済みモデルのロード
model = load_model(MODEL_PATH)

def set_servo_angle(angle):
    duty = 2.5 + (angle / 18.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)  # サーボモーターの動作待機

def set_dc_motor(direction, speed):
    if direction == "forward":
        GPIO.output(DC_A_PIN1, GPIO.HIGH)
        GPIO.output(DC_A_PIN2, GPIO.LOW)
        GPIO.output(DC_B_PIN1, GPIO.HIGH)
        GPIO.output(DC_B_PIN2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(DC_A_PIN1, GPIO.LOW)
        GPIO.output(DC_A_PIN2, GPIO.HIGH)
        GPIO.output(DC_B_PIN1, GPIO.LOW)
        GPIO.output(DC_B_PIN2, GPIO.HIGH)
    else:
        GPIO.output(DC_A_PIN1, GPIO.LOW)
        GPIO.output(DC_A_PIN2, GPIO.LOW)
        GPIO.output(DC_B_PIN1, GPIO.LOW)
        GPIO.output(DC_B_PIN2, GPIO.LOW)

    dc_motor_pwm.ChangeDutyCycle(speed)

def preprocess_image(image):
    # カメラ画像の前処理
    image = cv2.resize(image, (256, 256))  # 256x256にリサイズ
    image = image / 255.0  # 正規化
    return np.expand_dims(image, axis=0)

def predict_direction(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    direction = np.argmax(prediction)  # 最大の予測クラス
    return direction

def capture_images():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        image_path = f"{IMAGE_SAVE_PATH}/frame_{frame_count}.jpg"
        cv2.imwrite(image_path, frame)
        print(f"Image saved: {image_path}")
        frame_count += 1
        time.sleep(1)  # 1秒ごとに画像を保存

        direction = predict_direction(frame)
        if direction == 0:  # 直進
            set_dc_motor("forward", 30)
            set_servo_angle(90)  # サーボモーターを中央に設定
        elif direction == 1:  # 左折
            set_dc_motor("forward", 30)
            set_servo_angle(45)
        elif direction == 2:  # 右折
            set_dc_motor("forward", 30)
            set_servo_angle(135)

    cap.release()

def main():
    print("Press 's' to start, 'x' to stop, 'q' to quit.")
    running = False
    capture_thread = None

    try:
        while True:
            command = input("Enter command: ")
            if command == 's':
                if not running:
                    print("Starting...")
                    running = True
                    capture_thread = threading.Thread(target=capture_images)
                    capture_thread.start()

            elif command == 'x':
                if running:
                    print("Stopping...")
                    running = False
                    set_dc_motor("stop", 0)

            elif command == 'q':
                print("Exiting...")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        if capture_thread and capture_thread.is_alive():
            capture_thread.join()
        GPIO.cleanup()
        print("GPIO cleanup done.")

if __name__ == "__main__":
    main()

