import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence

# モデルをロード
MODEL_PATH = 'my_model2.h5'
model = load_model(MODEL_PATH)

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

    def __iter__(self):
        # イテレータオブジェクトを返すためのメソッドを実装
        for i in range(self.__len__()):
            yield self.__getitem__(i)

# テストデータ用ジェネレーター
test_gen = DataGenerator(test_image_dir, test_mask_dir, batch_size=4)

# モデルの評価
def evaluate_model(model, test_gen, num_images=5):
    loss, accuracy = model.evaluate(test_gen)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    
    # モデルの予測を表示
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

evaluate_model(model, test_gen)


