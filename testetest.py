import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image  # Pillowライブラリを追加

# モデルのロード
model = load_model('my_model.h5')

# 入力フォルダと出力フォルダ
input_folder = 'Pictures4'
output_folder = 'Outputs'
os.makedirs(output_folder, exist_ok=True)

# 入力フォルダ内のファイルを処理
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # ディレクトリや不要なファイルをスキップ
    if os.path.isdir(file_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Skipping: {file_path}")
        continue

    try:
        # 現在のファイルを表示
        print(f"Processing {filename}...")

        # 画像の読み込みと前処理
        image = load_img(file_path, target_size=(256, 256))
        image_array = img_to_array(image) / 255.0  # 正規化
        image_array = np.expand_dims(image_array, axis=0)

        # モデルによる予測
        prediction = model.predict(image_array)

        # 結果を保存
        output_path = os.path.join(output_folder, f"output_{filename}")
        prediction_image = (prediction[0] * 255).astype('uint8')  # スケールを戻す
        Image.fromarray(prediction_image.squeeze(), mode="L").save(output_path)

        print(f"Processed {filename}, saved to {output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

