import Jetson.GPIO as GPIO
import time

# GPIOピンの設定
MOTOR_PIN = 33  # PWM用GPIO

# GPIOの初期化
GPIO.setwarnings(False)  # 警告を無効化
GPIO.setmode(GPIO.BOARD)
GPIO.setup(MOTOR_PIN, GPIO.OUT)

# PWMの設定
pwm = GPIO.PWM(MOTOR_PIN, 500)  # 1kHzのPWM

def start_motor(speed=50):
    # モーターを前進させる
    pwm.start(speed)

def stop_motor():
    # モーターを停止させる
    pwm.ChangeDutyCycle(0)  # デューティサイクルを0に設定
    pwm.stop()

# メイン関数
def main():
    try:
        start_motor(50)  # モーターを50%のスピードで前進させる
        print("モーター動作中...")
        time.sleep(2)  # 2秒間動作
        stop_motor()   # モーターを停止
        print("モーター停止")
    except KeyboardInterrupt:
        print("停止します")
    finally:
        # GPIOリソースの解放
        pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()

