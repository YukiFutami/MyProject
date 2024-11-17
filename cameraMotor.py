import Jetson.GPIO as GPIO
import time
import subprocess
import cv2
import numpy as np

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
    # Set direction based on motor selection ('A' or 'B')
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
    
    # Control speed with PWM (0 to 100%)
    dc_motor_pwm.ChangeDutyCycle(speed)

# Initialize camera
cap = cv2.VideoCapture(0)  # Open default camera (0)

if not cap.isOpened():
    print("Error: Camera not detected")
    exit()

print("Press 'w' to move forward, 's' to move backward, 'a' to turn left, 'd' to turn right, 'p' to pause/unpause")

# Variable to track the pause state
paused = False

try:
    while True:
        # Capture frame-by-frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show the captured frame
        cv2.imshow('Camera Feed', frame)
        
        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('w') and not paused:  # Move forward
            set_dc_motor(50, "forward", "A")
            set_dc_motor(50, "forward", "B")
        elif key == ord('s') and not paused:  # Move backward
            set_dc_motor(50, "backward", "A")
            set_dc_motor(50, "backward", "B")
        elif key == ord('a') and not paused:  # Turn left
            set_dc_motor(50, "forward", "A")
            set_dc_motor(50, "backward", "B")
        elif key == ord('d') and not paused:  # Turn right
            set_dc_motor(50, "backward", "A")
            set_dc_motor(50, "forward", "B")
        elif key == ord('p'):  # Pause or unpause
            paused = not paused
            print("Paused" if paused else "Unpaused")
            time.sleep(1)  # Add a small delay to avoid rapid toggling of pause state
        elif key == ord('q'):  # Quit program
            break

finally:
    # Release the camera and cleanup GPIO
    cap.release()
    cv2.destroyAllWindows()
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
