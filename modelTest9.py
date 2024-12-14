// GPU using
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from tensorflow.keras.utils import Sequence

# バッチ処理用データジェネレーター
class DataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, target_size=(256, 256), shuffle=True):
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.image_filenames[k] for k in batch_indexes]
        batch_masks = [self.mask_filenames[k] for k in batch_indexes]
        
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
train_gen = DataGenerator(train_image_dir, train_mask_dir, batch_size=8, shuffle=True)  # バッチサイズを適切に設定
val_gen = DataGenerator(val_image_dir, val_mask_dir, batch_size=8, shuffle=True)

# モデル構築
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    
    # アップサンプリング層を適切に構成
    Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    Conv2D(1, (1, 1), activation='sigmoid')  # 出力形状を256x256x1に調整
])

# モデルコンパイル
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# モデルの訓練
history = model.fit(train_gen, validation_data=val_gen, epochs=20, batch_size=16, verbose=1)

# モデルの保存
model.save('my_model4.h5')
print("モデルの訓練と保存が完了しました！")

# 学習曲線の可視化
import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

