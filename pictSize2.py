import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 画像の読み込みとリサイズ
def load_images(image_dir, mask_dir, target_size=(256, 256)):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        print(f"Attempting to load image: {img_path}")  # 画像のパスを表示
        
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Error reading image {filename}. Skipping this file.")
            continue
        
        img_resized = cv2.resize(img, target_size)
        images.append(img_resized)
        print(f"Loaded image {filename}")
        
        # マスク画像の読み込み
        mask_filename = filename.replace('.png', '_mask.png')  # マスク画像名が同じ名前であると仮定
        mask_path = os.path.join(mask_dir, mask_filename)
        
        # マスク画像が存在するかチェック
        if not os.path.exists(mask_path):
            print(f"Mask image not found for {filename}. Skipping this file.")
            continue
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Error reading mask image {mask_filename}. Skipping this file.")
            continue
        
        mask_resized = cv2.resize(mask, target_size)
        # マスク画像の次元を (高さ, 幅, 1) に変換
        mask_resized = np.expand_dims(mask_resized, axis=-1)  # チャンネル次元を追加
        masks.append(mask_resized)
        print(f"Loaded mask for {filename}")
        
    print(f"Loaded {len(images)} images and {len(masks)} masks.")
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
def augment_and_save_data(X, y, output_image_dir, output_mask_dir, num_augmented=1000, batch_size=100):
    datagen = create_data_generator()

    # 訓練データ用の画像データ生成器
    image_datagen = datagen.flow(X, batch_size=batch_size, save_to_dir=output_image_dir, save_prefix='aug', save_format='png')
    mask_datagen = datagen.flow(y, batch_size=batch_size, save_to_dir=output_mask_dir, save_prefix='aug', save_format='png')

    # データ拡張を指定された枚数だけ実行
    num_batches = num_augmented // batch_size
    for _ in range(num_batches):
        next(image_datagen)  # 画像データの拡張
        next(mask_datagen)   # マスクデータの拡張
        print(f"Generated {batch_size} augmented images and masks.")

# 画像とマスクのディレクトリパス
image_dir = 'classified_color_images'  # 元の画像が保存されているディレクトリ
mask_dir = 'classified_color_images/masks2'  # マスク画像のディレクトリ

# 画像とマスクを読み込む
images, masks = load_images(image_dir, mask_dir)

# 出力先のディレクトリ設定（拡張後のデータ保存場所）
output_image_dir = 'augmented_images3'
output_mask_dir = 'augmented_masks3'

# 出力先ディレクトリが存在しない場合は作成
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)
    print(f"Created directory {output_image_dir}")
else:
    print(f"Directory {output_image_dir} already exists.")

if not os.path.exists(output_mask_dir):
    os.makedirs(output_mask_dir)
    print(f"Created directory {output_mask_dir}")
else:
    print(f"Directory {output_mask_dir} already exists.")

# データ拡張を実行して保存
augment_and_save_data(images, masks, output_image_dir, output_mask_dir, num_augmented=1000, batch_size=100)

