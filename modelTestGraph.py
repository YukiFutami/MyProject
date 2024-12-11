import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

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
image_dir = 'augmented_images'  # 拡張後の画像ディレクトリ
mask_dir = 'augmented_masks'    # 拡張後のマスクディレクトリ
images, masks = load_images(image_dir, mask_dir)
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# モデルの構築とコンパイル
model = build_unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 訓練過程をプロットする関数
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # 訓練と検証の損失をプロット
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 訓練と検証の精度をプロット
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 予測と結果を表示する関数
def predict_and_plot(model, X_val, y_val, idx=0):
    pred_mask = model.predict(np.expand_dims(X_val[idx], axis=0))
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    
    # 画像とマスク、予測結果を並べて表示
    plt.figure(figsize=(10, 5))
    
    # 元の画像
    plt.subplot(1, 3, 1)
    plt.imshow(X_val[idx])
    plt.title("Original Image")
    plt.axis('off')
    
    # 元のマスク（正解）
    plt.subplot(1, 3, 2)
    plt.imshow(y_val[idx], cmap='gray')
    plt.title("True Mask")
    plt.axis('off')
    
    # 予測したマスク
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask[0], cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
    
    plt.show()

# モデルの学習
history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_val, y_val))

# 学習履歴をプロット
plot_training_history(history)

# 学習後、予測結果を表示
predict_and_plot(model, X_val, y_val, idx=0)  # 0番目の画像で確認

# 検証
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

