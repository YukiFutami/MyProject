import os
import cv2

# ディレクトリのパス
base_folder = 'augmented_pictures'
output_folder = 'resized_images'
target_size = (256, 256)  # リサイズ後の画像サイズ

# 出力フォルダの作成
os.makedirs(output_folder, exist_ok=True)

# フォルダ内のすべての画像をリサイズして出力
for filename in os.listdir(base_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # 対象の拡張子を持つファイルのみ
        image_path = os.path.join(base_folder, filename)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, target_size)
        
        # リサイズ後の画像を保存
        resized_path = os.path.join(output_folder, filename)
        cv2.imwrite(resized_path, resized_image)

print(f"すべての画像が {target_size} のサイズにリサイズされ、{output_folder} に保存されました。")

