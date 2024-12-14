import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import Jetson.GPIO as GPIO
import time
import sys

# GPIO設定
SERVO_PIN = 32
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# サーボモーターのPWM設定
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz
pwm.start(7.5)  # 初期位置を90度に設定

# モデルのロード
model = load_model('my_model.h5')

# 入力フォルダと出力フォルダ
input_folder = 'Pictures4'
output_folder = 'Outputs'
os.makedirs(output_folder, exist_ok=True)

# 角度を設定する関数
def set_servo_angle(angle):
    duty_cycle = 2.5 + (angle / 18.0)  # 0度が2.5%、180度が12.5%
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)

# 入力フォルダ内のファイルを処理
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # ディレクトリや不要なファイルをスキップ
    if os.path.isdir(file_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Skipping: {file_path}")
        continue

    try:
        print(f"Processing {filename}...")

        # 画像の読み込みと前処理
        image = load_img(file_path, target_size=(256, 256))
        image_array = img_to_array(image) / 255.0  # 正規化
        image_array = np.expand_dims(image_array, axis=0)

        # モデルによる予測
        prediction = model.predict(image_array)
        prediction = prediction[0].squeeze()  # 必要に応じて予測結果を調整

        # 白線の位置に応じて角度を決定
        if np.mean(prediction) < 0.33:  # 白線が左にある
            angle = 45  # 左折
            print(f"Result: Left Turn (Angle: {angle})")
        elif np.mean(prediction) > 0.66:  # 白線が右にある
            angle = 135  # 右折
            print(f"Result: Right Turn (Angle: {angle})")
        else:  # 白線が中央にある
            angle = 90  # 直進
            print(f"Result: Straight (Angle: {angle})")

        # サーボモーターに角度を反映
        set_servo_angle(angle)

        # 結果を保存
        output_path = os.path.join(output_folder, f"output_{filename}")
        prediction_image = (prediction * 255).astype('uint8')  # スケールを戻す
        Image.fromarray(prediction_image, mode="L").save(output_path)
        print(f"Processed {filename}, saved to {output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# キーボード入力を待つ
try:
    while True:
        if input() == 'x':
            print("Exiting...")
            # サーボモーターを90度に戻す
            set_servo_angle(90)
            break
except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    pwm.stop()  # PWMを停止
    GPIO.cleanup()  # GPIOのリソースをクリーンアップ

