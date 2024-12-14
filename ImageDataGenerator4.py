import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 画像の読み込みとリサイズなし
def load_images(image_dir, target_size=None):
    images = []
    filenames = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image {filename}. Skipping this file.")
            continue
        
        # ファイル名を記録
        filenames.append(filename)
        images.append(img)
        
    return np.array(images), filenames

# データ拡張を適用して生成された画像を保存
def augment_and_save_data(X, filenames, output_image_dir, num_augmented=1000, batch_size=10):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.5, 1.5],
        channel_shift_range=30.0,
        fill_mode='nearest'
    )

    image_datagen = datagen.flow(X, batch_size=batch_size, seed=42)

    for i in range(num_augmented):
        img_batch = next(image_datagen)  # 拡張画像のバッチ
        
        # ファイル名を拡張した番号付きで生成
        base_name, ext = os.path.splitext(filenames[i % len(filenames)])
        aug_image_filename = f"{base_name}_aug{i + 1}{ext}"
        
        # 画像を保存
        img_path = os.path.join(output_image_dir, aug_image_filename)
        cv2.imwrite(img_path, img_batch[0])

        print(f"Saved augmented image: {img_path}")

# 画像のディレクトリパス
image_dir = 'classified_color_images'

# 画像を読み込む
images, filenames = load_images(image_dir)

# 出力先ディレクトリ設定
output_image_dir = 'augmented_images'
os.makedirs(output_image_dir, exist_ok=True)

# データ拡張を実行して保存
augment_and_save_data(images, filenames, output_image_dir, num_augmented=1000, batch_size=10)

