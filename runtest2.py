import os
import cv2
import numpy as np
import tensorflow as tf
import Jetson.GPIO as GPIO
from time import sleep, time

# TensorFlowのメモリ使用量を制限
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# GPIOピン設定
GPIO.setwarnings(False)
SERVO_PIN = 32
DC_PWM_PIN = 33
DC_A_PIN1 = 29
DC_A_PIN2 = 31
DC_B_PIN1 = 35
DC_B_PIN2 = 37

GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(DC_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_A_PIN1, GPIO.OUT)
GPIO.setup(DC_A_PIN2, GPIO.OUT)
GPIO.setup(DC_B_PIN1, GPIO.OUT)
GPIO.setup(DC_B_PIN2, GPIO.OUT)

servo = GPIO.PWM(SERVO_PIN, 50)
dc_motor = GPIO.PWM(DC_PWM_PIN, 100)
servo.start(7.5)
dc_motor.start(30)

model = tf.keras.models.load_model("my_model.h5")

def set_motor_direction(direction):
    if direction == "forward":
        GPIO.output(DC_A_PIN1, GPIO.HIGH)
        GPIO.output(DC_A_PIN2, GPIO.LOW)
        GPIO.output(DC_B_PIN1, GPIO.HIGH)
        GPIO.output(DC_B_PIN2, GPIO.LOW)
        servo.ChangeDutyCycle(7.5)
        print("Moving forward.")
    elif direction == "left":
        GPIO.output(DC_A_PIN1, GPIO.HIGH)
        GPIO.output(DC_A_PIN2, GPIO.LOW)
        GPIO.output(DC_B_PIN1, GPIO.HIGH)
        GPIO.output(DC_B_PIN2, GPIO.LOW)
        servo.ChangeDutyCycle(10.0)
        print("Turning left.")
    elif direction == "right":
        GPIO.output(DC_A_PIN1, GPIO.HIGH)
        GPIO.output(DC_A_PIN2, GPIO.LOW)
        GPIO.output(DC_B_PIN1, GPIO.HIGH)
        GPIO.output(DC_B_PIN2, GPIO.LOW)
        servo.ChangeDutyCycle(5.0)
        print("Turning right.")
    elif direction == "backward":
        GPIO.output(DC_A_PIN1, GPIO.LOW)
        GPIO.output(DC_A_PIN2, GPIO.HIGH)
        GPIO.output(DC_B_PIN1, GPIO.LOW)
        GPIO.output(DC_B_PIN2, GPIO.HIGH)
        servo.ChangeDutyCycle(7.5)
        print("Moving backward.")

def determine_direction(frame):
    try:
        img_array = cv2.resize(frame, (128, 128)) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        if predicted_class == 0:
            return "left"
        elif predicted_class == 1:
            return "forward"
        elif predicted_class == 2:
            return "right"
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def main():
    cap = cv2.VideoCapture(0)  # カメラのインデックスを指定して接続（0はデフォルトのカメラ）

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened. Press 's' to start, 'x' to stop.")

    is_running = False
    start_time = time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from camera.")
            break
        
        current_time = time() - start_time
        if not is_running or current_time > 5:  # 単純な時間制約で一時停止
            break

        direction = determine_direction(frame)
        if direction:
            print(f"Direction: {direction}")
            set_motor_direction(direction)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and not is_running:
            is_running = True
            print("Started.")
        elif key == ord('x'):
            break

        sleep(1)

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed. Program stopped.")

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

