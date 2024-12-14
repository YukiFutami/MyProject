import os
import cv2
import csv
import pandas as pd
import shutil
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 画像フォルダと出力先フォルダの設定
base_folder = 'resized_images'
unlabeled_folder = 'unlabeled_images'
labeled_folder = 'labeled_data'
auto_labeled_folder = 'auto_labeled_data'
output_csv = 'manual_labels.csv'
image_size = (256, 256)

# ラベル辞書
labels = {
    's': 'straight',  # 直進
    'l': 'left',      # 左折
    'r': 'right'      # 右折
}

def ensure_directories_exist():
    """必要なフォルダが存在しない場合は自動で作成する"""
    os.makedirs('resized_images', exist_ok=True)
    os.makedirs('unlabeled_images', exist_ok=True)
    os.makedirs('labeled_data/straight', exist_ok=True)
    os.makedirs('labeled_data/left', exist_ok=True)
    os.makedirs('labeled_data/right', exist_ok=True)
    os.makedirs('auto_labeled_data/straight', exist_ok=True)
    os.makedirs('auto_labeled_data/left', exist_ok=True)
    os.makedirs('auto_labeled_data/right', exist_ok=True)

def manual_labeling():
    images = [f for f in os.listdir(base_folder) if f.endswith(('.jpg', '.png'))][:50]  # 最初の50枚
    labeled_data = []

    for image_file in images:
        image_path = os.path.join(base_folder, image_file)
        image = cv2.imread(image_path)
        cv2.imshow("Image", image)

        print("Press 's' for Straight, 'l' for Left, 'r' for Right.")
        key = cv2.waitKey(0)

        # 無効なキー入力の場合に無視する
        if chr(key) in labels:
            labeled_data.append([image_file, labels[chr(key)]])
            print(f"Labeled {image_file} as {labels[chr(key)]}")
        elif chr(key) == 'q':  # 'q'で中断可能
            break
        else:
            print("Invalid key pressed. Please press 's', 'l', or 'r'.")

    cv2.destroyAllWindows()

    # ラベルをCSVに保存
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(labeled_data)

def auto_labeling():
    # 手動ラベルを読み込む
    if not os.path.exists(output_csv):
        print("Manual labels CSV does not exist.")
        return

    data = pd.read_csv(output_csv)

    # ラベル付きフォルダを作成
    for label in ['straight', 'left', 'right']:
        os.makedirs(os.path.join(labeled_folder, label), exist_ok=True)

    # ラベル付きデータをコピー
    for index, row in data.iterrows():
        src = os.path.join(base_folder, row['filename'])
        dest = os.path.join(labeled_folder, row['label'], row['filename'])
        shutil.copyfile(src, dest)

    # 残りのデータセットでラベル付けを行う
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        labeled_folder,
        target_size=image_size,
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    # 簡易モデルを構築
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # モデルを学習
    model.fit(train_gen, epochs=5)

    # 保存したモデルを使い、未分類画像を分類
    model.save('simple_direction_model.h5')

def auto_label_and_move_unlabeled():
    model = load_model('simple_direction_model.h5')

    # 未分類画像を予測して移動
    for image_file in os.listdir(unlabeled_folder):
        image_path = os.path.join(unlabeled_folder, image_file)
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, image_size)
        image_array = np.expand_dims(image_resized / 255.0, axis=0)

        prediction = model.predict(image_array)
        direction = np.argmax(prediction)

        label = ['straight', 'left', 'right'][direction]
        dest_path = os.path.join(auto_labeled_folder, label, image_file)
        shutil.copy(image_path, dest_path)

if __name__ == "__main__":
    # 必要なフォルダを自動で生成する
    ensure_directories_exist()
    
    # 1. 手動ラベル付け
    manual_labeling()
    
    # 2. 残りの画像を自動ラベル付け
    auto_labeling()
    
    # 3. 未分類画像を自動分類して移動
    auto_label_and_move_unlabeled()

