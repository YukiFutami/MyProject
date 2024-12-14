import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt  # 画像表示用ライブラリ

# デスクトップのパスを取得
desktop_path = os.path.expanduser("~/Desktop")

# 元の画像フォルダと保存先フォルダの指定
input_folder = 'classified_color_images'
output_folder = os.path.join(desktop_path, 'augmented_images')  # デスクトップに保存するフォルダ名

os.makedirs(output_folder, exist_ok=True)

# データ拡張の設定
datagen = ImageDataGenerator(
    rotation_range=30,      # ランダムに30度まで回転
    width_shift_range=0.2,  # 横方向にランダムにシフト
    height_shift_range=0.2, # 縦方向にランダムにシフト
    shear_range=0.2,        # ランダムなせん断
    zoom_range=0.2,         # ランダムなズーム
    horizontal_flip=True,   # 水平方向に反転
    fill_mode='nearest'     # 埋める際の手法
)

# 各クラスのフォルダごとに処理
for class_folder in os.listdir(input_folder):
    class_path = os.path.join(input_folder, class_folder)
    if not os.path.isdir(class_path):
        continue

    # 拡張画像の保存先フォルダ
    save_path = os.path.join(output_folder, class_folder)
    os.makedirs(save_path, exist_ok=True)

    # 各画像に対してデータ拡張を行う
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = load_img(img_path)  # 画像を読み込む
        x = img_to_array(img)    # 配列に変換
        x = x.reshape((1,) + x.shape)  # バッチ次元を追加

        # データ拡張の生成
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_path, save_prefix='aug', save_format='jpg'):
            i += 1
            if i >= 5:  # 各画像につき5枚生成
                break

        # デバッグのために1枚目の拡張画像を表示
        augmented_img = load_img(os.path.join(save_path, 'aug_0.jpg'))  # 生成された最初の画像をロード
        plt.imshow(augmented_img)
        plt.axis('off')  # 軸を非表示に
        plt.show()

