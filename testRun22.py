import cv2
import os
import time
import numpy as np
import tensorflow as tf
import Jetson.GPIO as GPIO

# Constants
GPIO.cleanup()  # GPIOをリセット
MODEL_PATH = './my_model.h5'
SAVE_DIR = './Picture5'
DIRECTION_FILE = 'direction.txt'

# Motor GPIO Pins
SERVO_PIN = 32
DC_PWM_PIN = 33
DC_A_PIN1 = 29
DC_A_PIN2 = 31
DC_B_PIN1 = 35
DC_B_PIN2 = 37
GPIO_MOTOR_SPEED = 50  # Example speed

# Configure GPIO pins
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(DC_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_A_PIN1, GPIO.OUT)
GPIO.setup(DC_A_PIN2, GPIO.OUT)
GPIO.setup(DC_B_PIN1, GPIO.OUT)
GPIO.setup(DC_B_PIN2, GPIO.OUT)

# PWM setup for servos and DC motors
servo_pwm = GPIO.PWM(SERVO_PIN, 50)
motor_pwm = GPIO.PWM(DC_PWM_PIN, 1000)
servo_pwm.start(7.5)  # Start servo at 90 degrees
motor_pwm.start(GPIO_MOTOR_SPEED)

# Ensure save directory exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error: Failed to load model. {e}")
    GPIO.cleanup()
    exit()

def save_frame(frame, counter):
    filename = os.path.join(SAVE_DIR, f'frame_{counter}.jpg')
    cv2.imwrite(filename, frame)

def control_gpio(direction):
    if direction == 'straight':
        servo_pwm.ChangeDutyCycle(7.5)
    elif direction == 'left':
        servo_pwm.ChangeDutyCycle(10)
    elif direction == 'right':
        servo_pwm.ChangeDutyCycle(5)

    GPIO.output(DC_A_PIN1, GPIO.HIGH)
    GPIO.output(DC_A_PIN2, GPIO.LOW)
    GPIO.output(DC_B_PIN1, GPIO.HIGH)
    GPIO.output(DC_B_PIN2, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(GPIO_MOTOR_SPEED)

def process_frame(frame):
    frame_resized = cv2.resize(frame, (256, 256))
    frame_normalized = frame_resized / 255.0
    frame_input = np.expand_dims(frame_normalized, axis=0)

    predictions = model.predict(frame_input)
    directions = ['straight', 'left', 'right']
    direction = directions[np.argmax(predictions[0])]

    with open(DIRECTION_FILE, 'w') as file:
        file.write(direction)

    control_gpio(direction)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    is_running = False
    frame_counter = 0

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                print("Starting processing...")
                is_running = True
            elif key == ord('x'):
                print("Stopping processing...")
                is_running = False

            if is_running:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame")
                    break

                frame_counter += 1
                save_frame(frame, frame_counter)
                process_frame(frame)
                time.sleep(1)  # 1-second interval between frame processing
            else:
                time.sleep(0.1)  # Avoid busy-waiting
    finally:
        cap.release()
        cv2.destroyAllWindows()
        servo_pwm.stop()
        motor_pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()

