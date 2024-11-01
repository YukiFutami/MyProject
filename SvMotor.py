## 先生のコード　模写

import Jetson.GPIO as GPIO
import time
import subprocess ## シェルコマンドをpythonから実行するためのモジュール

## Shell commandを実行する関数
sudo_password = ""

def run_command(command):
    full_command = f"echo {sudo_password} | sudo -S {command}"
    subprocess.run(full_command, shell=True, check=True)

## busyboxのインストール確認
## busyboxとは　多機能なUNIXコマンドを集めたツールセット
## インストールされてない場合、自動的にインストールする
try:
    subprocess.run("busybox --help", shell=True, check=True)
    print("busybox is already installed.")
except subprocess.CalledProcessError:
    print("busybox not found. Installing...")
    run_command("apy update && apt install -y busybox")

## devmemコマンドの実行
## 特定のメモリアドレスに直接アクセスして値を読み書きするためのコマンド
## ここでJetson nanoのハードウェア設定を行っている　ピンのモード変更
commands = [
    "busybox devmem 0x700031fc 32 0x45",
    "busybox devmem 0x6000d504 32 0x2",
    "busybox devmem 0x70003248 32 0x46",
    "busybox devmem 0x6000d100 32 0x00",
] 

for command in commands:
    run_command(command)

## GPIOのセットアップ
servo_pin = 7 # サーボモーターを接続するGPIOピンを指定、ここでは7番を使用
 
GPIO.setmode(GPIO.BOARD) ## 物理ピン番号でピンを指定
GPIO.setup(servo_pin, GPIO.OUT) ## 指定したピンを出力モードにする

## PWMの設定
## サーボモーターの回転角度を制御するための信号を生成
servo = GPIO.PWM(servo_pin, 50) ## 周波数を５０Hzに設定
servo.start(0)

## サーボの角度を設定する関数
## サーボモーターの角度を0~180に設定
def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18) ## 計算式で角度に対するデューティ比を求める
    servo.ChangeDutyCycle(duty_cycle) 
    time.sleep(0.5) ## サーボが設定した角度に到達するまで0.5秒待機する
    servo.ChangeDutyCycle(0) ## 信号をオフに四、モーターの不要な振動を防ぐ

## メイン処理
try:
    while True: ## 角度入力を無限ループで待ち受ける
        inputValue = int(input("put angle value : ")) ## 角度を取得しその値を
        set_servo_angle(inputValue) ## ここに渡す
finally: ## 例外が発生してもPWM信号を停止し、GPIOの設定をクリアする
    servo.stop()
    GPIO.cleanup()