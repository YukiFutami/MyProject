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

# 画像の正規化 (0～1の範囲にスケーリング)
def normalize_image(image):
    return image / 255.0

# ノイズ除去（ガウシアンブラー）
def remove_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# データ拡張（反転、回転）
def augment_image(image):
    # 水平方向に反転
    flipped_img = cv2.flip(image, 1)

    # 回転（ランダム回転）
    rows, cols = image.shape[:2]
    angle = random.randint(-15, 15)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_img = cv2.warpAffine(image, M, (cols, rows))

    return rotated_img

# 画像の前処理をまとめた関数
def preprocess_images(input_dir, output_dir, target_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            # リサイズ
            img = resize_image(img, target_size)

            # グレースケール変換
            img = convert_to_grayscale(img)

            # 正規化
            img = normalize_image(img)

            # ノイズ除去
            img = remove_noise(img)

            # データ拡張（例：反転、回転）
            img = augment_image(img)

            # 保存（正規化後に元のスケールに戻す）
            img = (img * 255).astype(np.uint8)  # 正規化した値を元のスケールに戻す
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)
            print(f"Processed and saved: {output_path}")

# 実行例
input_dir = "/path/to/input_images"  # 入力画像が保存されているディレクトリ
output_dir = "/path/to/output_images"  # 出力画像を保存するディレクトリ

preprocess_images(input_dir, output_dir)
