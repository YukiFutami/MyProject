import Jetson.GPIO as GPIO
import time

# GPIO設定
MOTOR_PWM = 33  # PWM用ピン
MOTOR_A = 29    # モーターA制御ピン
MOTOR_B = 35    # モーターB制御ピン
GPIO.setmode(GPIO.BOARD)
GPIO.setup(MOTOR_PWM, GPIO.OUT)
GPIO.setup(MOTOR_A, GPIO.OUT)
GPIO.setup(MOTOR_B, GPIO.OUT)

# PWMインスタンス生成 (初期周波数100Hz)
pwm = GPIO.PWM(MOTOR_PWM, 100)
pwm.start(0)  # 初期デューティ比は0%

try:
    # モーターを前進させる
    GPIO.output(MOTOR_A, GPIO.HIGH)
    GPIO.output(MOTOR_B, GPIO.LOW)

    # 周波数をテスト
    for freq in [100, 500, 1000, 2000]:
        print(f"周波数を {freq} Hz に設定")
        pwm.ChangeFrequency(freq)
        pwm.ChangeDutyCycle(50)  # デューティ比50%
        time.sleep(2)  # 2秒間動作

    # モーター停止
    pwm.ChangeDutyCycle(0)
    GPIO.output(MOTOR_A, GPIO.LOW)
    GPIO.output(MOTOR_B, GPIO.LOW)

except KeyboardInterrupt:
    print("中断されました")

finally:
    pwm.stop()
    GPIO.cleanup()

