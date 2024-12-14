import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ディレクトリとファイルのパス
resize_images_path = 'resized_images/'
labels_csv_path = 'labels.csv'
auto_labeled_results_csv_path = 'auto_labeled_results.csv'
dataset_path = 'dataset/'

# メッセージを出力する関数
def log_message(message):
    print(f"[INFO] {message}")

# ディレクトリの作成
def create_directories():
    log_message("Creating dataset directory structure...")
    for folder in ['train', 'validation', 'test']:
        for label in ['0', '1', '2']:
            dir_path = os.path.join(dataset_path, folder, label)
            os.makedirs(dir_path, exist_ok=True)
    log_message("Directory structure created successfully.")

# ファイルのコピー
def copy_labeled_images():
    log_message("Copying labeled images to respective class directories...")
    labels_df = pd.read_csv(labels_csv_path)
    auto_labeled_df = pd.read_csv(auto_labeled_results_csv_path)
    
    for index, row in labels_df.iterrows():
        src_path = os.path.join(resize_images_path, row['image'])
        dest_path = os.path.join(dataset_path, 'train', str(row['label']))
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            log_message(f"Image {row['image']} not found in {resize_images_path}.")

    for index, row in auto_labeled_df.iterrows():
        src_path = os.path.join(resize_images_path, row['image'])
        dest_path = os.path.join(dataset_path, 'train', str(row['label']))
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            log_message(f"Auto-labeled image {row['image']} not found in {resize_images_path}.")
    log_message("Image copying completed.")

# データセットの分割
def split_dataset():
    log_message("Splitting dataset into train, validation, and test sets...")
    all_images = []
    all_labels = []

    for label_dir in ['0', '1', '2']:
        for phase in ['train', 'test']:
            path = os.path.join(dataset_path, phase, label_dir)
            for img_name in os.listdir(path):
                all_images.append(os.path.join(path, img_name))
                all_labels.append(int(label_dir))

    X_train_val, X_test, y_train_val, y_test = train_test_split(all_images, all_labels, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    log_message("Creating dataset splits...")
    for img_path, label in zip(X_train, y_train):
        dst_dir = os.path.join(dataset_path, 'train', str(label))
        shutil.move(img_path, os.path.join(dst_dir, os.path.basename(img_path)))

    for img_path, label in zip(X_val, y_val):
        dst_dir = os.path.join(dataset_path, 'validation', str(label))
        shutil.move(img_path, os.path.join(dst_dir, os.path.basename(img_path)))

    for img_path, label in zip(X_test, y_test):
        dst_dir = os.path.join(dataset_path, 'test', str(label))
        shutil.move(img_path, os.path.join(dst_dir, os.path.basename(img_path)))
    
    log_message("Dataset split completed.")

# 実行の流れ
def main():
    log_message("Starting the dataset preparation process...")
    create_directories()
    copy_labeled_images()
    split_dataset()
    log_message("Dataset preparation process completed successfully.")

if __name__ == "__main__":
    main()

