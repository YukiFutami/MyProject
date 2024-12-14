import os
import cv2
import pandas as pd
import shutil

# データの準備
base_folder = 'self_data'
labeled_folder = 'labeled_data'
manual_labels_csv = 'manual_labels.csv' # ここを正しいファイル名に変更

# 手動ラベルファイルが存在しない場合は、手動ラベルCSVを自動生成
if not os.path.isfile(manual_labels_csv):
    # 新たなラベルCSVファイルを作成する
    columns = ['filename', 'label']
    df = pd.DataFrame(columns=columns)
    df.to_csv(manual_labels_csv, index=False)
    print(f"手動ラベルファイル '{manual_labels_csv}' が生成されました。")

# 手動ラベルを読み込み
try:
    data = pd.read_csv(manual_labels_csv)
except FileNotFoundError:
    print(f"Error: {manual_labels_csv} not found.")

# 新たにラベル付けされたデータをコピー
for label in ['straight', 'left', 'right']:
    os.makedirs(os.path.join(labeled_folder, label), exist_ok=True)

for index, row in data.iterrows():
    src = os.path.join(base_folder, row['filename'])
    dest = os.path.join(labeled_folder, row['label'], row['filename'])
    shutil.copyfile(src, dest)

    # 画像の表示処理を追加
    image_path = os.path.join(labeled_folder, row['label'], row['filename'])
    image = cv2.imread(image_path)

    if image is not None:
        cv2.imshow('Labeling Image', image)
        key = cv2.waitKey(0)  # キーを押すまで画像を表示し続ける

        if key == ord('s'):
            row['label'] = 'straight'
        elif key == ord('l'):
            row['label'] = 'left'
        elif key == ord('r'):
            row['label'] = 'right'
        
        # 手動ラベルをCSVファイルに追加
        data.to_csv(manual_labels_csv, index=False)

        cv2.destroyAllWindows()
    else:
        print(f"画像の読み込みに失敗しました: {image_path}")

print(f"手動ラベル付けが完了しました。{labeled_folder} に「straight」、「left」、「right」のフォルダが作成されました。")

