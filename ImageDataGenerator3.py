import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


# 元の画像フォルダと保存先フォルダの指定
input_folder = 'classified_color_images'
output_folder = 'augmented_images'

os.makedirs(output_folder, exist_ok=True)

# データ拡張の設定
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 各画像に対してデータ拡張を行う
for img_file in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_file)
    if not os.path.isfile(img_path):
        continue

    img = load_img(img_path)  # 画像を読み込む
    x = img_to_array(img)    # 配列に変換
    x = x.reshape((1,) + x.shape)  # バッチ次元を追加

    # 保存先フォルダ
    save_path = os.path.join(output_folder, os.path.splitext(img_file)[0])
    os.makedirs(save_path, exist_ok=True)

    # データ拡張の生成と保存
    for i, batch in enumerate(datagen.flow(x, batch_size=1, save_to_dir=save_path, save_prefix='aug', save_format='png')):
        if i >= 5:  # 各画像につき5枚生成
            break

        # 拡張画像の名前を変更（元のファイル名に番号を付ける）
        new_file_name = f"{os.path.splitext(img_file)[0]}_{i}.png"
        os.rename(os.path.join(save_path, 'aug_0.png'), os.path.join(save_path, new_file_name))

