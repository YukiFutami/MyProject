import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# パラメータ
IMAGE_SIZE = (128, 128)  # モデルに合わせるための画像サイズ
CSV_FILE = "labels.csv"
IMAGE_FOLDER = "resized_images"  # 画像が保存されているフォルダ
AUTO_LABEL_CSV = "auto_labeled_results.csv"  # 自動ラベル結果の保存先

# labels.csvの準備
def initialize_csv():
    if not os.path.exists(CSV_FILE):
        print(f"{CSV_FILE} が存在しません。新しいファイルを作成します。")
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ファイル名", "ラベル"])
        print(f"{CSV_FILE} を作成しました。手動でラベルを追加してください。")
    else:
        print(f"{CSV_FILE} が既に存在します。続行します。")

# 手動ラベリング部分
def manual_labeling():
    existing_labels = pd.read_csv(CSV_FILE)
    labeled_files = set(existing_labels["ファイル名"])

    for image_file in os.listdir(IMAGE_FOLDER):
        if image_file.endswith(".png") or image_file.endswith(".jpg"):
            if image_file not in labeled_files:
                print(f"画像: {image_file}")
                label = input("ラベルを入力してください (0: 直進, 1: 左折, 2: 右折): ")

                while label not in ["0", "1", "2"]:
                    label = input("無効な入力です。再度入力してください (0: 直進, 1: 左折, 2: 右折): ")

                with open(CSV_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([image_file, label])
                print(f"{image_file} にラベル {label} を追加しました。")

# モデル構築
def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# データの準備
def load_data():
    data = pd.read_csv(CSV_FILE)
    images = []
    labels = []

    for _, row in data.iterrows():
        img_path = os.path.join(IMAGE_FOLDER, row["ファイル名"])
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(int(row["ラベル"]))

    return np.array(images), np.array(labels)

# 自動ラベリング
def auto_label_unlabeled(model):
    data = pd.read_csv(CSV_FILE)
    labeled_files = set(data["ファイル名"])
    predictions = []

    for image_file in os.listdir(IMAGE_FOLDER):
        if image_file not in labeled_files:
            img_path = os.path.join(IMAGE_FOLDER, image_file)
            img = load_img(img_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            label = np.argmax(prediction)
            predictions.append([image_file, label])

    # 自動ラベルを保存
    with open(AUTO_LABEL_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ファイル名", "ラベル"])
        writer.writerows(predictions)
    print(f"自動ラベリングが完了しました。結果は {AUTO_LABEL_CSV} に保存されました。")

# 実行フロー
if __name__ == "__main__":
    # 初期化
    initialize_csv()

    # 手動ラベリング
    manual_labeling()

    # モデル学習
    images, labels = load_data()
    if len(labels) > 0:
        model = build_model()
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

        # 自動ラベリング
        auto_label_unlabeled(model)
    else:
        print("ラベル付きデータが不足しています。手動でラベリングを増やしてください。")

