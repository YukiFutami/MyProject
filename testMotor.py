import Jetson.GPIO as GPIO
import time
import random  # 模擬的にモデルの指示をランダムに生成

# GPIO設定
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

# GPIO初期化
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1_A, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_A, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1_B, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2_B, GPIO.OUT)
GPIO.setup(servo_pin, GPIO.OUT)

# PWM初期化
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # PWM周波数: 200Hz
servo_pwm = GPIO.PWM(servo_pin, 50)  # サーボモーターは50Hz

dc_motor_pwm.start(0)  # DCモーターを停止状態でスタート
servo_pwm.start(7.5)   # サーボモーターをニュートラル位置（90度）でスタート

def set_dc_motor(speed=50, direction="forward"):
    """DCモーターの動作を制御（両モーター）"""
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1_A, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2_A, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin1_B, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2_B, GPIO.LOW)
    elif direction == "stop":
        GPIO.output(dc_motor_dir_pin1_A, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2_A, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin1_B, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2_B, GPIO.LOW)
    else:
        print("Invalid direction")
        return
    print(f"DCモーター: direction={direction}, speed={speed}")
    dc_motor_pwm.ChangeDutyCycle(speed)

def set_servo_angle(angle):
    """サーボモーターの角度を制御"""
    duty_cycle = 2.5 + (angle / 18.0)  # 角度をデューティ比に変換
    print(f"サーボモーター: angle={angle}, duty_cycle={duty_cycle}")
    servo_pwm.ChangeDutyCycle(duty_cycle)

try:
    print("DCモーターが常に前進中...")
    set_dc_motor(50, "forward")  # DCモーターを前進で開始
    
    while True:
        # ランダムにモデルの指示を生成
        model_result = random.choice(["left", "right", "straight"])
        print(f"Model direction: {model_result}")

        # サーボモーターの動作
        if model_result == "left":
            set_servo_angle(45)  # 左方向
        elif model_result == "right":
            set_servo_angle(135)  # 右方向
        elif model_result == "straight":
            set_servo_angle(90)  # 直進

        time.sleep(1)  # 1秒待機して次のモデル結果を模擬

except KeyboardInterrupt:
    print("停止中...")
finally:
    # 全モーター停止
    set_dc_motor(0, "stop")
    servo_pwm.ChangeDutyCycle(0)
    dc_motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()

