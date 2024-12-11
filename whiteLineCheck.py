import os
import cv2
import numpy as np

# 白線の有無を分類する関数（例：しきい値処理や既存モデル）
def is_white_line_present(image_gray, threshold=100):
    """
    グレースケール画像を入力し、白線の有無を判定
    :param image_gray: グレースケール画像
    :param threshold: 白線と判定するピクセルのしきい値
    :return: 白線がある場合True、ない場合False
    """
    white_pixel_ratio = np.mean(image_gray > threshold)  # しきい値以上のピクセル割合
    return white_pixel_ratio > 0.05  # 白線が一定割合以上存在するか

# 分類と保存処理
def classify_and_save_images(input_color_dir, output_color_dir, output_gray_dir):
    """
    カラー画像とグレースケール画像を「白線あり」と判定された場合に保存
    :param input_color_dir: 元画像（カラー）の入力ディレクトリ
    :param output_color_dir: 白線ありのカラー画像保存ディレクトリ
    :param output_gray_dir: 白線ありのグレースケール画像保存ディレクトリ
    """
    os.makedirs(output_color_dir, exist_ok=True)
    os.makedirs(output_gray_dir, exist_ok=True)

    for filename in os.listdir(input_color_dir):
        filepath = os.path.join(input_color_dir, filename)
        img_color = cv2.imread(filepath)

        if img_color is not None:
            # グレースケール変換
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            # 白線の有無を判定
            if is_white_line_present(img_gray):
                # 白線ありと判定された場合、保存
                cv2.imwrite(os.path.join(output_color_dir, filename), img_color)
                gray_filename = filename.replace('.jpg', '_gray.png')  # グレースケール保存用
                cv2.imwrite(os.path.join(output_gray_dir, gray_filename), img_gray)

# 入力・出力ディレクトリの指定
input_color_dir = 'Pictures4'  # 元のカラー画像フォルダ
output_color_dir = 'classified_color_images'  # 白線ありのカラー画像保存先
output_gray_dir = 'classified_gray_images'   # 白線ありのグレースケール画像保存先

# 実行
classify_and_save_images(input_color_dir, output_color_dir, output_gray_dir)

