import Jetson.GPIO as GPIO
import time
import subprocess
import cv2
import numpy as np
import threading  # To run the background task

# Set the sudo password as a variable for easy updating
sudo_password = "your_password_here"

# Function to run shell commands with the sudo password
def run_command(command):
    full_command = f"echo {sudo_password} | sudo -S {command}"
    subprocess.run(full_command, shell=True, check=True)

# Check if busybox is installed; if not, install it
try:
    subprocess.run("busybox --help", shell=True, check=True)
    print("busybox is already installed.")
except subprocess.CalledProcessError:
    print("busybox not found. Installing...")
    run_command("apt update && apt install -y busybox")

# Define devmem commands
commands = [
    "busybox devmem 0x700031fc 32 0x45",
    "busybox devmem 0x6000d504 32 0x2",
    "busybox devmem 0x70003248 32 0x46",
    "busybox devmem 0x6000d100 32 0x00"
]

# Execute each devmem command
for command in commands:
    run_command(command)

# Set up GPIO pins for servo and DC motor control
servo_pin = 32                # PWM-capable pin for servo motor
dc_motor_pwm_pin = 33         # Unified PWM pin for both DC motors' speed
dc_motor_dir_pin1_A = 29      # Direction control pin 1 for DC motor A
dc_motor_dir_pin2_A = 31      # Direction control pin 2 for DC motor A
dc_motor_dir_pin1_B = 35      # Direction control pin 1 for DC motor B
dc_motor_dir_pin2_B = 37      # Direction control pin 2 for DC motor B

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1_A, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_A, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1_B, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_B, GPIO.OUT)

# Configure PWM on servo and unified DC motor pin
servo = GPIO.PWM(servo_pin, 50)  # 50Hz for servo motor
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 1kHz for DC motors
servo.start(0)
dc_motor_pwm.start(0)

# Function to set servo angle
def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)

# Function to set DC motor speed and direction for Motor A or B
def set_dc_motor(speed, direction, motor="A"):
    if motor == "A":
        if direction == "forward":
            GPIO.output(dc_motor_dir_pin1_A, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2_A, GPIO.LOW)
        elif direction == "backward":
            GPIO.output(dc_motor_dir_pin1_A, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2_A, GPIO.HIGH)
    elif motor == "B":
        if direction == "forward":
            GPIO.output(dc_motor_dir_pin1_B, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2_B, GPIO.LOW)
        elif direction == "backward":
            GPIO.output(dc_motor_dir_pin1_B, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2_B, GPIO.HIGH)
    
    dc_motor_pwm.ChangeDutyCycle(speed)

# Initialize camera
cap = cv2.VideoCapture(0)  # Open default camera (0)

if not cap.isOpened():
    print("Error: Camera not detected")
    exit()

# Variable to track the pause state
paused = False

# Function to save images from the camera feed in the background
def save_images_from_camera():
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_filename = f"captured_image_{frame_count}.png"
        cv2.imwrite(image_filename, frame)
        print(f"Image saved: {image_filename}")
        frame_count += 1
        time.sleep(1)

# Start the background thread for saving images
image_thread = threading.Thread(target=save_images_from_camera)
image_thread.daemon = True
image_thread.start()

# Move functions
def move_forward():
    set_dc_motor(50, "forward", "A")
    set_dc_motor(50, "forward", "B")

def move_backward():
    set_dc_motor(50, "backward", "A")
    set_dc_motor(50, "backward", "B")

def turn_left():
    set_dc_motor(50, "forward", "A")
    set_dc_motor(50, "backward", "B")

def turn_right():
    set_dc_motor(50, "backward", "A")
    set_dc_motor(50, "forward", "B")

print("Press 'w' to move forward, 's' to move backward, 'a' to turn left, 'd' to turn right, 'p' to pause/unpause")

try:
    while True:
        key = cv2.waitKey(10) & 0xFF  # Modified wait time for key input
        
        if key == ord('w') and not paused:
            move_forward()
        elif key == ord('s') and not paused:
            move_backward()
        elif key == ord('a') and not paused:
            turn_left()
        elif key == ord('d') and not paused:
            turn_right()
        elif key == ord('p'):  # Toggle pause state
            paused = not paused
            print("Paused" if paused else "Unpaused")
        elif key == ord('q'):
            break
        
        time.sleep(0.1)  # Short delay to prevent CPU overload

finally:
    cap.release()
    cv2.destroyAllWindows()
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
