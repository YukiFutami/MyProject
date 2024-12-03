# モデル構築

import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# 画像とマスクのロード関数
def load_images(image_dir, mask_dir, target_size=(256, 256)):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, filename))
        img_resized = cv2.resize(img, target_size)
        images.append(img_resized)
        
        mask_filename = filename.replace('.jpg', '_mask.png')
        mask = cv2.imread(os.path.join(mask_dir, mask_filename), cv2.IMREAD_GRAYSCALE)
        mask_resized = cv2.resize(mask, target_size)
        masks.append(mask_resized)
        
    return np.array(images), np.array(masks)

# U-Netモデルの構築
def build_unet_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    # Output layer
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = models.Model(inputs, output)
    return model

# 訓練データと検証データの準備
image_dir = 'path_to_images'
mask_dir = 'path_to_masks'
images, masks = load_images(image_dir, mask_dir)
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# モデルの構築とコンパイル
model = build_unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# モデルの学習
history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_val, y_val))

# 検証
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
