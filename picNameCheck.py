import os

# 出力先ディレクトリを指定
output_image_dir = 'augmented_images4'
output_mask_dir = 'augmented_masks4'


# 拡張後の画像とマスクの対応を確認
augmented_images = sorted(os.listdir(output_image_dir))
augmented_masks = sorted(os.listdir(output_mask_dir))

# 一致しているか確認
for img_name, mask_name in zip(augmented_images, augmented_masks):
    if img_name.replace('.png', '_mask.png') != mask_name:
        print(f"Mismatch: {img_name} <-> {mask_name}")
    else:
        print(f"Matched: {img_name} <-> {mask_name}")

