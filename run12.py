import os
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
    elif direction == "left":
        GPIO.output(DC_A_PIN1, GPIO.HIGH)
        GPIO.output(DC_A_PIN2, GPIO.LOW)
        GPIO.output(DC_B_PIN1, GPIO.HIGH)
        GPIO.output(DC_B_PIN2, GPIO.LOW)
        servo.ChangeDutyCycle(10.0)
    elif direction == "right":
        GPIO.output(DC_A_PIN1, GPIO.HIGH)
        GPIO.output(DC_A_PIN2, GPIO.LOW)
        GPIO.output(DC_B_PIN1, GPIO.HIGH)
        GPIO.output(DC_B_PIN2, GPIO.LOW)
        servo.ChangeDutyCycle(5.0)

def determine_direction(image_path):
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))  # 画像の読み込みサイズを128x128に制限
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
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
        print(f"Error processing image {image_path}: {e}")
        return None

def main():
    image_folder = "Picture"
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in folder.")
        return
    
    print(f"Found {len(image_files)} images. Processing...")
    
    start_time = time()
    
    while True:
        current_time = time() - start_time
        if current_time > len(image_files):
            break
        
        image_file = image_files[int(current_time)]
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing {image_path}...")
        
        direction = determine_direction(image_path)
        if direction:
            print(f"Direction: {direction}")
            set_motor_direction(direction)
        sleep(1)

    print("Processing completed.")

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

