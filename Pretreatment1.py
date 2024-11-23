import cv2
import os
import numpy as np
import random

# 画像のリサイズ
def resize_image(image, target_size=(224, 224)):
    return cv2.resize(image, target_size)

# グレースケール変換
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ノイズ除去（ガウシアンブラー）
def remove_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# データ拡張（回転）
def augment_image(image):
    rows, cols = image.shape[:2]
    angle = random.randint(-15, 15)  # ランダム回転角度
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_img = cv2.warpAffine(image, M, (cols, rows))
    return rotated_img

# 画像のクロップ（上部3分の1を削除）
def crop_image(image):
    height, width = image.shape[:2]
    cropped_img = image[height // 3 :, :]  # 上から3分の1をカット
    return cropped_img

# 画像の前処理
def preprocess_images(input_dir, output_dir, target_size=(224, 224), use_grayscale=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_counter = 1  # ファイル名用カウンター

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            # 読み込み失敗時はスキップ
            if img is None:
                print(f"Failed to load image: {img_path}, skipping.")
                continue

            # クロップ（上部3分の1を削除）
            img = crop_image(img)

            # リサイズ
            img = resize_image(img, target_size)

            # グレースケール変換（必要に応じて）
            if use_grayscale:
                img = convert_to_grayscale(img)

            # ノイズ除去
            img = remove_noise(img)

            # 元画像を保存
            output_path = os.path.join(output_dir, f"image_{file_counter:04d}_original.jpg")
            cv2.imwrite(output_path, img)
            print(f"Saved original: {output_path}")

            # データ拡張（例: 回転画像）
            augmented_img = augment_image(img)
            augmented_path = os.path.join(output_dir, f"image_{file_counter:04d}_augmented.jpg")
            cv2.imwrite(augmented_path, augmented_img)
            print(f"Saved augmented: {augmented_path}")

            file_counter += 1  # 次のファイル名に進む

# 実行例
input_dir = "/path/to/input_images"  # 入力画像ディレクトリ
output_dir = "/path/to/output_images"  # 出力画像ディレクトリ

# グレースケールを使いたい場合は `use_grayscale=True`
preprocess_images(input_dir, output_dir, target_size=(224, 224), use_grayscale=False)
