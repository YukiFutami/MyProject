import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence

# モデルをロード
MODEL_PATH = 'my_model2.h5'
model = load_model(MODEL_PATH)

# トレーニングデータディレクトリ
train_image_dir = 'train1/images'
train_mask_dir = 'train1/masks'
# 検証データディレクトリ
val_image_dir = 'val1/images'
val_mask_dir = 'val1/masks'
# テストデータディレクトリ
test_image_dir = 'test1/images'
test_mask_dir = 'test1/masks'

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

# トレーニング、検証、テストデータ用ジェネレーター
train_gen = DataGenerator(train_image_dir, train_mask_dir, batch_size=4)
val_gen = DataGenerator(val_image_dir, val_mask_dir, batch_size=4)
test_gen = DataGenerator(test_image_dir, test_mask_dir, batch_size=4)

# モデルの学習と評価
history = model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=10
)

# モデルの学習履歴を可視化
def plot_history(history):
    if history.history:
        # 損失と精度の推移を描画
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss during training')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy during training')

        plt.show()
    else:
        print("Error: history object is None or invalid. Skipping plot.")

plot_history(history)

# モデルの評価結果を可視化
def plot_predictions(model, test_gen, num_images=5):
    for i in range(num_images):
        images, true_masks = next(test_gen)
        pred_masks = model.predict(images)
        
        # マスクを復元
        true_mask = np.squeeze(true_masks[i])
        pred_mask = np.squeeze(pred_masks[i])

        # 可視化
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(true_mask, cmap='gray')
        axes[0].set_title('True Mask')
        axes[0].axis('off')

        axes[1].imshow(pred_mask, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')

        plt.show()

plot_predictions(model, test_gen)

