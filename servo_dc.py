import Jetson.GPIO as GPIO
import time
from pynput import keyboard

# GPIOの設定
GPIO.cleanup()  # GPIOをリセット
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# サーボモーターのピン設定
servo_pin = 32  # サーボモーターのPWMピン

# DCモーターのピン設定
dc_motor_pwm_pin = 33         # PWM信号で両方のモーターの速度を制御
dc_motor_dir_pin1_A = 29      # モーターAの方向制御ピン1
dc_motor_dir_pin2_A = 31      # モーターAの方向制御ピン2
dc_motor_dir_pin1_B = 35      # モーターBの方向制御ピン1
dc_motor_dir_pin2_B = 37      # モーターBの方向制御ピン2

# サーボモーターのセットアップ
GPIO.setup(servo_pin, GPIO.OUT)
servo = GPIO.PWM(servo_pin, 50)  # 50HzでPWM信号を送る（サーボの一般的な周波数）
servo.start(7.5)  # 初期位置90°に設定

# DCモーターのセットアップ
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1_A, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_A, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1_B, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_B, GPIO.OUT)

dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 周波数1kHzでPWM制御
dc_motor_pwm.start(0)  # PWMを0%で開始

# サーボモーターを指定した角度に動かす関数
def set_servo_angle(angle):
    duty_cycle = max(2, min(12, 2 + (angle / 18)))  # 角度に応じてデューティサイクルを設定
    print(f"Setting servo to {angle}° with duty cycle: {duty_cycle}%")
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # モーターが動く時間を確保
    servo.ChangeDutyCycle(0)  # PWM信号を停止

# DCモーターの動作制御関数
def set_dc_motor(speed=0, direction="forward", motor="A"):  # Modified here for forward only
    if motor == "A" or motor == "B":
        if direction == "forward":
            GPIO.output(dc_motor_dir_pin1_A if motor == "A" else dc_motor_dir_pin1_B, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2_A if motor == "A" else dc_motor_dir_pin2_B, GPIO.LOW)
        elif direction == "stop":
            GPIO.output(dc_motor_dir_pin1_A if motor == "A" else dc_motor_dir_pin1_B, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2_A if motor == "A" else dc_motor_dir_pin2_B, GPIO.LOW)
    dc_motor_pwm.ChangeDutyCycle(speed)  # PWMで速度を設定

# 両方のDCモーターを制御する関数
def set_both_dc_motors(speed=0, direction="forward"):  # Modified here for forward only
    set_dc_motor(speed, direction, "A")
    set_dc_motor(speed, direction, "B")

# キー入力に応じてサーボとDCモーターを制御する関数
def on_press(key):
    try:
        if key.char == 'j':  # 'j'キーでサーボを135°に設定
            set_servo_angle(125)
        elif key.char == 'k':  # 'k'キーでサーボを50°に設定
            set_servo_angle(60)
        elif key.char == 'r':  # 'r'キーでサーボを90°に戻す
            set_servo_angle(90)
        elif key.char == 'w':  # 'w'キーで前進
            set_both_dc_motors(30, "forward")
        elif key.char == 'x':  # 'x'キーで停止
            set_both_dc_motors(0, "stop")
    except AttributeError:
        pass

# キーリスナーを開始
listener = keyboard.Listener(on_press=on_press)
listener.start()

# プログラムが終了しないように待機
try:
    while True:
        time.sleep(0.1)  # 負荷軽減のため100ms待機
except KeyboardInterrupt:
    print("\nProgram interrupted.")
finally:
    listener.stop()  # キーリスナーを停止
    servo.stop()  # サーボのPWMを停止
    dc_motor_pwm.stop()  # DCモーターのPWMを停止
    GPIO.cleanup()  # GPIOをクリーンアップ

