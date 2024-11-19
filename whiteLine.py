import cv2
import numpy as np

# 白線抽出関数
def extract_white_lines(image):
    # 画像がグレースケールの場合、そのまま使用
    # もしカラー画像の場合は、まずグレースケールに変換
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 白色の範囲を指定（例えば、閾値範囲：180以上のピクセル値を白とする）
    lower_white = np.array([180, 180, 180])  # RGBでの白の下限
    upper_white = np.array([255, 255, 255])  # RGBでの白の上限
    
    # 白線をマスクする
    mask = cv2.inRange(image, lower_white, upper_white)
    
    # マスクを適用して白線部分を取り出す
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result

# 使用例
image_path = 'input_image.png'
image = cv2.imread(image_path)
white_lines = extract_white_lines(image)

# 結果を表示
cv2.imshow('White Lines', white_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
