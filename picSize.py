import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# データ拡張の設定
def create_data_generator():
    datagen = ImageDataGenerator(
        rotation_range=30,          # ランダム回転 (±30度)
        width_shift_range=0.2,      # ランダムな水平方向のシフト
        height_shift_range=0.2,     # ランダムな垂直方向のシフト
        shear_range=0.2,            # ランダムなせん断変換
        zoom_range=0.2,             # ランダムなズーム
        horizontal_flip=True,       # 水平反転
        brightness_range=[0.5, 1.5],# 明るさのランダム調整
        channel_shift_range=30.0,   # 彩度（色相）の変化
        fill_mode='nearest'         # 新しいピクセルの色の設定
    )
    return datagen

# データ拡張を適用して画像を保存
def augment_and_save_data(image_dir, output_dir, num_augmented=1000):
    """
    :param image_dir: 拡張元の画像ディレクトリ
    :param output_dir: 拡張後の画像保存先
    :param num_augmented: 拡張する画像の総数
    """
    datagen = create_data_generator()
    os.makedirs(output_dir, exist_ok=True)
    
    images = []
    for filename in os.listdir(image_dir):
        filepath = os.path.join(image_dir, filename)
        img = cv2.imread(filepath)
        if img is not None:
            images.append(img)
    
    images = np.array(images) / 255.0  # 正規化
    
    # データ拡張実行
    generator = datagen.flow(images, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpg')
    for _ in range(num_augmented):
        next(generator)

# 実行するディレクトリを指定
classified_image_dir = 'classified_color_images'  # 分類後の画像保存ディレクトリ
output_dir = 'augmented_images'            # データ拡張後の保存先
augment_and_save_data(classified_image_dir, output_dir, num_augmented=1000)

