import cv2
import numpy as np
import os

# 画像の読み込みと白線検出を行う関数
def create_mask_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像読み込みエラー: {image_path}")
        return None
    
    img_resized = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return mask

# 元画像のディレクトリパス
image_dir = 'classified_color_images'

# ディレクトリが存在しない場合はエラーを出す
if not os.path.exists(image_dir):
    print(f"指定された画像ディレクトリ {image_dir} が見つかりません。")
    exit()

# マスク画像を保存するディレクトリ
mask_dir = os.path.join(image_dir, 'masks')

# 出力先ディレクトリが存在しない場合は作成
os.makedirs(mask_dir, exist_ok=True)

# ディレクトリ内のファイル一覧を表示
print("ディレクトリ内のファイル一覧:", os.listdir(image_dir))

# 画像ごとにマスクを作成し保存
for filename in os.listdir(image_dir):
    print(f"ファイルを見つけました: {filename}")
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # 他の拡張子にも対応
        image_path = os.path.join(image_dir, filename)
        print(f"画像パス: {image_path}")

        # マスク画像を作成
        mask = create_mask_image(image_path)
        if mask is None:
            continue

        # マスク画像を保存
        mask_filename = filename.replace('.jpg', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_filename)

        try:
            success = cv2.imwrite(mask_path, mask)
            if success:
                print(f"マスク画像を保存しました: {mask_path}")
            else:
                print(f"マスク画像の保存に失敗しました: {mask_path}")
        except Exception as e:
            print(f"エラーが発生しました: {e}")

