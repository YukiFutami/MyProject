import os
import cv2
import csv
import shutil
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# フォルダ設定
base_folder = 'augmented_pictures'
output_csv = 'labeled_data.csv'
unlabeled_folder = 'unlabeled_images'
labeled_folder = 'labeled_images'
auto_labeled_folder = 'auto_labeled_images'
questionable_folder = 'questionable_images'

# ラベル付けに使用するキー
labels = {'s': 'straight', 'l': 'left', 'r': 'right'}

def manual_labeling():
    images = [f for f in os.listdir(base_folder) if f.endswith(('.jpg', '.png'))][:50]  # 最初の50枚
    labeled_data = []

    for image_file in images:
        image_path = os.path.join(base_folder, image_file)
        image = cv2.imread(image_path)
        cv2.imshow("Image", image)

        print("Press 's' for Straight, 'l' for Left, 'r' for Right, 'd' to delete.")
        key = cv2.waitKey(0)

        if chr(key) in labels:
            labeled_data.append([image_file, labels[chr(key)]])
            print(f"Labeled {image_file} as {labels[chr(key)]}")
        elif chr(key) == 'd':  # 'd'で削除
            os.remove(image_path)
            print(f"Deleted {image_file}")
        elif chr(key) == 'q':  # 'q'で中断可能
            break

    cv2.destroyAllWindows()

    # ラベルをCSVに保存
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(labeled_data)

def auto_labeling():
    # 手動ラベルを読み込む
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
        target_size=(256, 256),
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
    for image_file in os.listdir(unlabeled_folder):
        image_path = os.path.join(unlabeled_folder, image_file)
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (256, 256))
        image_array = np.expand_dims(image_resized / 255.0, axis=0)

        prediction = model.predict(image_array)
        max_index = np.argmax(prediction)

        # 確信がない場合に「疑わしい画像」フォルダに移動
        if np.max(prediction) < 0.7:  # 予測確率が低い場合
            dest_path = os.path.join(questionable_folder, image_file)
            shutil.copy(image_path, dest_path)
            print(f"Moved {image_file} to questionable_images folder.")
        else:
            direction = ['straight', 'left', 'right'][max_index]
            dest_path = os.path.join(auto_labeled_folder, direction, image_file)
            shutil.copy(image_path, dest_path)

# 主導ラベル付けを実行
manual_labeling()

# 自動ラベル付けを実行
auto_labeling()

