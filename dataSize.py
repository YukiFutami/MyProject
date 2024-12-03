# データ拡張

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 画像の読み込みとリサイズ
def load_images(image_dir, mask_dir, target_size=(256, 256)):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, filename))
        img_resized = cv2.resize(img, target_size)
        images.append(img_resized)
        
        # マスク画像の読み込み
        mask_filename = filename.replace('.jpg', '_mask.png')  # マスク画像名が同じ名前であると仮定
        mask = cv2.imread(os.path.join(mask_dir, mask_filename), cv2.IMREAD_GRAYSCALE)
        mask_resized = cv2.resize(mask, target_size)
        masks.append(mask_resized)
        
    return np.array(images), np.array(masks)

# データ拡張を行うImageDataGeneratorの設定
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
        fill_mode='nearest'         # 画像を切り取った際に新しいピクセルの色を設定
    )
    return datagen

# データ拡張を適用して生成された画像を保存
def augment_and_save_data(X, y, output_image_dir, output_mask_dir, num_augmented=1000):
    datagen = create_data_generator()

    # 訓練データ用の画像データ生成器
    image_datagen = datagen.flow(X, batch_size=1, save_to_dir=output_image_dir, save_prefix='aug', save_format='jpg')
    mask_datagen = datagen.flow(y, batch_size=1, save_to_dir=output_mask_dir, save_prefix='aug', save_format='png')

    # データ拡張を指定された枚数だけ実行
    for _ in range(num_augmented):
        next(image_datagen)  # 画像データの拡張
        next(mask_datagen)   # マスクデータの拡張

# 画像とマスクのディレクトリパス
image_dir = 'path_to_images'
mask_dir = 'path_to_masks'

# 画像とマスクを読み込む
images, masks = load_images(image_dir, mask_dir)

# 出力先のディレクトリ設定（拡張後のデータ保存場所）
output_image_dir = 'augmented_images'
output_mask_dir = 'augmented_masks'

# 出力先ディレクトリが存在しない場合は作成
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# データ拡張を実行して保存
augment_and_save_data(images, masks, output_image_dir, output_mask_dir, num_augmented=1000)
