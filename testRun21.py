import cv2
import os
import time
import numpy as np
import tensorflow as tf
import Jetson.GPIO as GPIO  # Use Jetson.GPIO for GPIO control on Jetson Nano

# Constants
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

# 以前のPWMチャンネルが存在する場合（もしあれば）アンエクスポート
servo_pwm.stop()
servo_pwm.unexport()

# Configure GPIO pins
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)  # Disable GPIO warnings
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(DC_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_A_PIN1, GPIO.OUT)
GPIO.setup(DC_A_PIN2, GPIO.OUT)
GPIO.setup(DC_B_PIN1, GPIO.OUT)
GPIO.setup(DC_B_PIN2, GPIO.OUT)

# PWM setup for servos and DC motors
servo_pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz for servo PWM frequency
motor_pwm = GPIO.PWM(DC_PWM_PIN, 1000)  # 1kHz for DC motor PWM

servo_pwm.start(7.5)  # Start servo at 90 degrees
motor_pwm.start(GPIO_MOTOR_SPEED)  # Start motor at specified speed

# Ensure save directory exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def save_frame(frame, counter):
    """Save captured frame to the specified directory."""
    filename = os.path.join(SAVE_DIR, f'frame_{counter}.jpg')
    cv2.imwrite(filename, frame)

def read_direction():
    """Read the direction from a file."""
    if os.path.exists(DIRECTION_FILE):
        with open(DIRECTION_FILE, 'r') as file:
            return file.readline().strip()
    return None

def control_gpio(direction):
    """Control GPIO pins for motors based on the direction."""
    if direction == 'straight':
        # Set servo to 90 degrees (straight)
        print("Control GPIO: Moving straight, servo at 90 degrees")
        servo_pwm.ChangeDutyCycle(7.5)
    elif direction == 'left':
        # Set servo to some angle for left turn
        print("Control GPIO: Turning left, servo angle adjusted")
        servo_pwm.ChangeDutyCycle(10)  # Adjust duty cycle for left turn
    elif direction == 'right':
        # Set servo to some angle for right turn
        print("Control GPIO: Turning right, servo angle adjusted")
        servo_pwm.ChangeDutyCycle(5)   # Adjust duty cycle for right turn

    # Set DC motors to constant speed
    GPIO.output(DC_A_PIN1, GPIO.HIGH)
    GPIO.output(DC_A_PIN2, GPIO.LOW)
    GPIO.output(DC_B_PIN1, GPIO.HIGH)
    GPIO.output(DC_B_PIN2, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(GPIO_MOTOR_SPEED)
    print(f"Control GPIO: DC motor speed set to {GPIO_MOTOR_SPEED}")

def process_frame(frame):
    """Process a captured frame and control GPIO based on predictions."""
    # Preprocess frame for the model
    frame_resized = cv2.resize(frame, (256, 256))
    frame_normalized = frame_resized / 255.0
    frame_input = np.expand_dims(frame_normalized, axis=0)

    # Model prediction
    predictions = model.predict(frame_input)
    if predictions[0][0] > 0.5:  # Example threshold for classification
        direction = 'straight'
    elif predictions[0][1] > 0.5:
        direction = 'left'
    elif predictions[0][2] > 0.5:
        direction = 'right'
    else:
        direction = 'straight'  # Default to straight if unsure

    # Save direction to file
    with open(DIRECTION_FILE, 'w') as file:
        file.write(direction)

    print(f"Model output: {direction}")
    control_gpio(direction)

def main():
    """Main function to capture video frames and process them."""
    cap = cv2.VideoCapture(0)  # Open the default camera

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera opened. Press 's' to start, 'x' to stop.")
    frame_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break

            # Save frame every second
            frame_counter += 1
            save_frame(frame, frame_counter)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                print("Start processing images...")
                process_frame(frame)
            elif key == ord('x'):
                print("Emergency stop")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        servo_pwm.stop()
        motor_pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()

