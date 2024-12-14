import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
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
            img_resized = cv2.resize(img, self.target_size)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0  # マスクを正規化
            mask_resized = cv2.resize(mask, self.target_size)
            mask_resized = np.expand_dims(mask_resized, axis=-1)  # チャンネル次元を追加
            
            images.append(img_resized)
            masks.append(mask_resized)
        
        return np.array(images), np.array(masks)

# ディレクトリパス
train_image_dir = 'train1/images'
train_mask_dir = 'train1/masks'
val_image_dir = 'val1/images'
val_mask_dir = 'val1/masks'

# ジェネレーターのインスタンスを作成
train_gen = DataGenerator(train_image_dir, train_mask_dir, batch_size=4)  # バッチサイズを小さく
val_gen = DataGenerator(val_image_dir, val_mask_dir, batch_size=4)

# 既存の学習済みモデルを読み込む
model = load_model('my_model3.h5')

# モデルの再訓練
model.fit(train_gen, validation_data=val_gen, epochs=10)

# 再訓練したモデルを保存
model.save('my_model_retrained.h5')
print("モデルの再訓練が完了し、保存されました！")

