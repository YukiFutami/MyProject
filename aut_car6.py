import time
import cv2
import numpy as np
from keras.models import load_model
import Jetson.GPIO as GPIO
from pynput import keyboard
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
servo_pin = 32
GPIO.setup(servo_pin, GPIO.OUT)
servo = GPIO.PWM(servo_pin, 50)
servo.start(7.5)  # 90度

# DCモーターの設定
dc_motor_pwm_pin = 33
dc_motor_dir_pin1_A = 29
dc_motor_dir_pin2_A = 31
dc_motor_dir_pin1_B = 35
dc_motor_dir_pin2_B = 37

GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1_A, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_A, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1_B, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_B, GPIO.OUT)

dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)
dc_motor_pwm.start(0)

# サーボモーターを角度に応じて動かす
def set_servo_angle(angle):
    duty_cycle = max(2, min(12, 2 + (angle / 18)))
    print(f"Setting servo to {angle}° with duty cycle: {duty_cycle}%")
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)

# DCモーター制御
def set_dc_motor(speed, direction):
    pwm_value = max(10, min(100, speed))  # スピードを10〜100の範囲で制限
    GPIO.output(dc_motor_dir_pin1_A, GPIO.HIGH if direction == "forward" else GPIO.LOW)
    GPIO.output(dc_motor_dir_pin2_A, GPIO.LOW if direction == "forward" else GPIO.HIGH)
    GPIO.output(dc_motor_dir_pin1_B, GPIO.HIGH if direction == "forward" else GPIO.LOW)
    GPIO.output(dc_motor_dir_pin2_B, GPIO.LOW if direction == "forward" else GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(pwm_value)

# モーター停止
def stop_dc_motor():
    GPIO.output(dc_motor_dir_pin1_A, GPIO.LOW)
    GPIO.output(dc_motor_dir_pin2_A, GPIO.LOW)
    GPIO.output(dc_motor_dir_pin1_B, GPIO.LOW)
    GPIO.output(dc_motor_dir_pin2_B, GPIO.LOW)
    dc_motor_pwm.ChangeDutyCycle(0)

# 白線認識用モデル
line_model = load_model('my_model3.h5')
direction_model = load_model('trained_model.h5')  # 方向モデルをロード

# 状態管理
running = False

# キーボードイベントのリスナー
def on_press(key):
    global running
    try:
        if key.char == 's':
            print("Starting motion...")
            running = True
        elif key.char == 'x':
            print("Stopping motion...")
            running = False
            stop_dc_motor()
    except AttributeError:
        pass

# カメラフレームを処理
def process_camera_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    print("Camera started. Press 's' to start and 'x' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        cv2.imshow("Frame", frame)

        if running:
            # 前処理
            resized_frame = cv2.resize(frame, (256, 256)) / 255.0
            input_data = np.expand_dims(resized_frame, axis=0)

            # 方向予測
            prediction = direction_model.predict(input_data)
            action = np.argmax(prediction)  # 0: 直進, 1: 左折, 2: 右折

            if action == 0:  # 直進
                print("Action: Straight")
                set_servo_angle(90)
            elif action == 1:  # 左折
                print("Action: Left")
                set_servo_angle(60)
            elif action == 2:  # 右折
                print("Action: Right")
                set_servo_angle(120)

            # モーターを動作
            set_dc_motor(50, "forward")  # スピードを30に制限
        else:
            stop_dc_motor()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# メイン処理
if __name__ == "__main__":
    # キーボードリスナーを開始
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        process_camera_frame()
    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        stop_dc_motor()
        servo.stop()
        GPIO.cleanup()

