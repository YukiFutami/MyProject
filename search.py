import os
import pandas as pd

# フォルダ内のファイル名リスト
resize_images = set(os.listdir('resized_images'))

# CSVから画像名リストを取得
labels = pd.read_csv('labels.csv')
labeled_images = set(labels['image'].tolist())

# 差分を確認
missing_images = labeled_images - resize_images
print("Missing images:", missing_images)

