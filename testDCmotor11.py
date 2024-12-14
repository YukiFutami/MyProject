import time
import Jetson.GPIO as GPIO

# GPIO設定
DC_MOTOR_FORWARD_PIN = 32  # DCモーター前方向のピン番号（仮設定）
START_KEY = 's'

# GPIOの設定
GPIO.setmode(GPIO.BOARD)
GPIO.setup(DC_MOTOR_FORWARD_PIN, GPIO.OUT)

# スタートメッセージ
def start_message():
    print("Press 's' to start...")

# DCモーターを前方向に回転
def start_dc_motor():
    GPIO.output(DC_MOTOR_FORWARD_PIN, GPIO.HIGH)
    print("DC Motor is running forward.")

# メインループ
def main():
    start_message()
    
    while True:
        key_input = input()  # キー入力を待つ
        if key_input == START_KEY:
            start_dc_motor()
            break

# プログラム開始
if __name__ == "__main__":
    main()

