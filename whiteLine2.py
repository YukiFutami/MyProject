import cv2
import os

# 白線抽出関数（グレースケール画像用）
def extract_white_lines_from_grayscale(image):
    # 白線の閾値設定（180以上を白線と見なす）
    _, mask = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    return mask

# フォルダ内の画像を処理する関数
def process_images(input_folder, output_folder):
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 入力フォルダ内の画像を順番に処理
    for filename in os.listdir(input_folder):
        # 画像ファイルかどうかを確認
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 入力画像のパス
            input_path = os.path.join(input_folder, filename)
            
            # グレースケール画像として読み込み
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to load image: {input_path}")
                continue
            
            # 白線を抽出
            white_lines = extract_white_lines_from_grayscale(image)
            
            # 出力画像のパス
            output_path = os.path.join(output_folder, filename)
            
            # 抽出結果を保存
            cv2.imwrite(output_path, white_lines)
            print(f"Processed and saved: {output_path}")

# 使用例
input_folder = "processed_pictures"  # 入力画像フォルダ
output_folder = "white_lines_output"  # 出力フォルダ

process_images(input_folder, output_folder)

