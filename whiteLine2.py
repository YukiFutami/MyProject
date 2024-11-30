# グレースケールから結果を保存するコード
import cv2

# 白線抽出関数
def extract_white_lines_from_grayscale(image):
    # 白線の閾値設定（180以上を白線と見なす）
    _, mask = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    return mask

# 使用例
image_path = 'grayscale_image.png'  # グレースケール画像のパス
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # グレースケール画像を直接読み込み

# 白線を抽出
white_lines = extract_white_lines_from_grayscale(image)

# 結果を保存
cv2.imwrite('white_lines_output.png', white_lines)

# 結果を表示
cv2.imshow('White Lines', white_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
