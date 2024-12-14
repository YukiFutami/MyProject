import os
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import shutil

# フォルダ設定
base_folder = 'resized_images'
unlabeled_folder = 'unlabeled_images'
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

def auto_label_and_move_unlabeled():
    # 手動ラベルを読み込む
    if not os.path.exists(output_csv):
        print("Manual labels CSV does not exist.")
        return

    data = pd.read_csv(output_csv)

    # 未分類画像を予測して分類し、auto_labeled_folderに移動
    model = load_model('simple_direction_model.h5')

    for image_file in os.listdir(unlabeled_folder):
        image_path = os.path.join(unlabeled_folder, image_file)
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, image_size)
        image_array = np.expand_dims(image_resized / 255.0, axis=0)

        prediction = model.predict(image_array)
        direction = np.argmax(prediction)

        label = data.loc[data['filename'] == image_file, 'label'].values[0]  # 手動ラベルから取得
        dest_path = os.path.join(auto_labeled_folder, label, image_file)
        shutil.copy(image_path, dest_path)

if __name__ == "__main__":
    # 必要なフォルダを自動で生成する
    ensure_directories_exist()
    
    # 未分類画像を自動分類して移動
    auto_label_and_move_unlabeled()

