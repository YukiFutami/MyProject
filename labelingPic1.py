import os
import cv2
import csv
import pandas as pd
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# データの準備
base_folder = 'augmented_pictures'
labeled_folder = 'labeled_data'
unlabeled_folder = 'unlabeled_images'
manual_labels_csv = 'labeled_images.csv'
output_folder = 'auto_labeled_data'

# 手動ラベルを読み込み
data = pd.read_csv(manual_labels_csv)

# 新たにラベル付けされたデータをコピー
for label in ['straight', 'left', 'right']:
    labeled_path = os.path.join(labeled_folder, label)
    if not os.path.exists(labeled_path):
        os.makedirs(labeled_path)

for index, row in data.iterrows():
    src = os.path.join(base_folder, row['filename'])
    dest = os.path.join(labeled_folder, row['label'], row['filename'])
    shutil.copyfile(src, dest)

# 未分類フォルダの作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
model.fit(train_gen, epochs=5)

# 新しいモデルを保存
model.save('simple_direction_model.h5')

# 未分類フォルダのラベル付け
unlabeled_files = [f for f in os.listdir(unlabeled_folder) if os.path.isfile(os.path.join(unlabeled_folder, f))]
for filename in unlabeled_files:
    file_path = os.path.join(unlabeled_folder, filename)
    img = cv2.imread(file_path)
    img = cv2.resize(img, (256, 256)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)

    if max(prediction[0]) < 0.8:  # 決定的な予測がつかない場合
        cv2.imwrite(os.path.join(output_folder, filename), img * 255)
    else:
        label = np.argmax(prediction)
        label_name = train_gen.class_indices.keys()[label]
        target_path = os.path.join(labeled_folder, label_name, filename)
        shutil.move(file_path, target_path)

