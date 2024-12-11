import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from tensorflow.keras.utils import Sequence

# バッチ処理用データジェネレーター
class DataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, target_size=(256, 256)):
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        batch_images = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        batch_masks = self.mask_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        
        images = []
        masks = []
        for img_name, mask_name in zip(batch_images, batch_masks):
            img_path = os.path.join(self.image_dir, img_name)
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            img = cv2.imread(img_path) / 255.0  # 画像を正規化
            img = cv2.resize(img, self.target_size)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0  # マスクを正規化
            mask = cv2.resize(mask, self.target_size)
            mask = np.expand_dims(mask, axis=-1)  # チャンネル次元を追加
            
            images.append(img)
            masks.append(mask)
        
        return np.array(images), np.array(masks)

# ディレクトリパス
train_image_dir = 'train/images'
train_mask_dir = 'train/masks'
val_image_dir = 'val/images'
val_mask_dir = 'val/masks'

# ジェネレーターのインスタンスを作成
train_gen = DataGenerator(train_image_dir, train_mask_dir, batch_size=8)
val_gen = DataGenerator(val_image_dir, val_mask_dir, batch_size=8)


# モデル構築
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    
    # Conv2DTranspose 層を使用してアップサンプリング
    Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),  # 120x120 -> 240x240
    Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),   # 240x240 -> 480x480
    
    # 最終的に256x256に収束させるために適切なサイズに縮小
    Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),   # 480x480 -> 256x256
    
    # 最終的に256x256に縮小
    Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same', activation='sigmoid')  # 出力サイズ: 256x256
])

# モデルコンパイル
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# モデルの訓練
model.fit(train_gen, validation_data=val_gen, epochs=10)

print("モデルの訓練が完了しました！")

