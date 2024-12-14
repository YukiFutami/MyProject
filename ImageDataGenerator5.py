import os
import cv2

# 元の画像フォルダと新しい画像フォルダの設定
input_folder = 'classified_color_images'
output_folder = 'augmented_pictures'
os.makedirs(output_folder, exist_ok=True)

# 画像ファイルをリスト化
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 左右反転処理
def flip_horizontal(image_path):
    image = cv2.imread(image_path)
    flipped = cv2.flip(image, 1)  # 1: 左右反転
    return flipped

# 各画像について反転処理を行い、新しいフォルダに保存
for image_file in image_files:
    input_path = os.path.join(input_folder, image_file)
    flipped_image = flip_horizontal(input_path)
    
    # 元の画像名を使用して保存
    base_name, ext = os.path.splitext(image_file)
    output_name = f'{base_name}_flipped{ext}'
    output_path = os.path.join(output_folder, output_name)
    cv2.imwrite(output_path, flipped_image)

